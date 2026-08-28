"""Auditable strategy readiness registry, dashboard, and fail-closed DEMO gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import BASE_DIR, SETTINGS, Settings


DEMO_READY = "DEMO_READY"
VALID_STATUSES = {
    "RESEARCHING", "REJECTED", "BACKTEST_PASS", "SHADOW_READY",
    "DEMO_FORWARD", DEMO_READY, "REVOKED",
}


class StrategyDeploymentBlocked(PermissionError):
    pass


def strategy_configuration_hash(settings: Settings, strategy_mode: str) -> str:
    digest = hashlib.sha256()
    digest.update(strategy_mode.encode("utf-8"))
    if strategy_mode == "calibrated":
        if not settings.decision_policy_path.is_file():
            raise StrategyDeploymentBlocked(
                f"decision policy is missing: {settings.decision_policy_path}"
            )
        digest.update(settings.decision_policy_path.read_bytes())
        digest.update((BASE_DIR / "decision_engine.py").read_bytes())
        digest.update((BASE_DIR / "asset_predictive_engine.py").read_bytes())
    elif strategy_mode == "liquidity_breakout":
        digest.update((BASE_DIR / "liquidity_breakout.py").read_bytes())
        configuration = {
            "minimum_rrr": settings.liquidity_min_rrr,
            "minimum_touches": settings.liquidity_min_touches,
            "volume_expansion": settings.liquidity_volume_expansion,
            "momentum_body_fraction": settings.liquidity_momentum_body_fraction,
            "maximum_trades_per_day": settings.liquidity_max_trades_per_day,
        }
        digest.update(json.dumps(configuration, sort_keys=True).encode("utf-8"))
    else:
        raise StrategyDeploymentBlocked(f"unsupported strategy mode: {strategy_mode}")
    return digest.hexdigest()


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return float(value) if value else None


def _parse_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _reconciliation_proof(settings: Settings) -> dict[str, Any]:
    marker = _load_json(settings.report_dir / "reconciliation_status.json", {})
    synced_at = _parse_utc(marker.get("synced_at"))
    now = datetime.now(timezone.utc)
    fresh = bool(
        synced_at
        and timedelta(0) <= now - synced_at <= timedelta(minutes=settings.readiness_max_age_minutes)
    )
    valid = bool(
        marker.get("status") == "SUCCESS"
        and marker.get("require_demo_account") is True
        and marker.get("trade_mode") == 0
        and fresh
    )
    return {
        "status": marker.get("status", "MISSING"),
        "synced_at": marker.get("synced_at"),
        "server": marker.get("server"),
        "trade_mode": marker.get("trade_mode"),
        "require_demo_account": marker.get("require_demo_account"),
        "orders": marker.get("orders"),
        "deals": marker.get("deals"),
        "maximum_age_minutes": settings.readiness_max_age_minutes,
        "fresh_demo_reconciliation": valid,
    }


def _calibrated_entry(settings: Settings, reconciliation: dict[str, Any]) -> dict[str, Any]:
    policy = _load_json(settings.decision_policy_path, {"policies": [], "approval_audit": []})
    policies = {str(row["symbol"]): row for row in policy.get("policies", [])}
    evidence = {
        str(row["symbol"]): row
        for row in _load_csv(settings.report_dir / "forward_evidence.csv")
    }
    symbols: list[dict[str, Any]] = []
    for symbol in settings.symbols:
        active = policies.get(symbol, {})
        row = evidence.get(symbol, {})
        sample = int(row.get("sample_size") or 0)
        sessions = int(row.get("sessions") or 0)
        expectancy = _float(row, "expectancy_after_costs")
        profit_factor = _float(row, "profit_factor")
        drawdown_pct = _float(row, "max_drawdown_pct")
        blockers: list[str] = []
        if not reconciliation["fresh_demo_reconciliation"]:
            blockers.append("FRESH_DEMO_RECONCILIATION_REQUIRED")
        if not active.get("enabled", False):
            blockers.append("POLICY_DISABLED")
        if not active.get("approved", False):
            blockers.append("POLICY_NOT_APPROVED")
        if sample < 30:
            blockers.append(f"FORWARD_TRADES_{sample}_OF_30")
        if sessions < 10:
            blockers.append(f"FORWARD_SESSIONS_{sessions}_OF_10")
        if expectancy is None or expectancy <= 0:
            blockers.append("NON_POSITIVE_FORWARD_EXPECTANCY")
        if profit_factor is None or profit_factor < 1.20:
            blockers.append("FORWARD_PROFIT_FACTOR_BELOW_1_20")
        if drawdown_pct is None:
            blockers.append("FORWARD_DRAWDOWN_PERCENT_UNAVAILABLE")
        elif drawdown_pct > 5.0:
            blockers.append("FORWARD_DRAWDOWN_ABOVE_5_PERCENT")
        symbols.append(
            {
                "symbol": symbol,
                "metrics": {
                    "trades": sample,
                    "sessions": sessions,
                    "net_profit_after_costs": _float(row, "net_profit_after_costs"),
                    "expectancy_after_costs": expectancy,
                    "profit_factor": profit_factor,
                    "max_drawdown_pct": drawdown_pct,
                },
                "blockers": blockers,
            }
        )
    rejected = any(
        "REJECTION" in str(row.get("action", "")).upper()
        for row in policy.get("approval_audit", [])
    )
    blockers = [
        f"{row['symbol']}:{blocker}"
        for row in symbols for blocker in row["blockers"]
    ]
    status = DEMO_READY if not blockers else ("REVOKED" if rejected else "REJECTED")
    return {
        "strategy_id": "asset_calibrated",
        "display_name": "BTC and Gold Asset Calibrated",
        "strategy_mode": "calibrated",
        "status": status,
        "demo_deployable": status == DEMO_READY,
        "configuration_hash": strategy_configuration_hash(settings, "calibrated"),
        "symbols": symbols,
        "blockers": blockers,
    }


def _liquidity_entry(settings: Settings) -> dict[str, Any]:
    search = _load_json(settings.report_dir / "liquidity_parameter_search.json", {"symbols": {}})
    backtest = _load_json(settings.report_dir / "liquidity_backtest.json", {"summaries": []})
    portfolio = next(
        (row for row in backtest.get("summaries", []) if row.get("symbol") == "PORTFOLIO"),
        {},
    )
    symbols: list[dict[str, Any]] = []
    for symbol in settings.symbols:
        selection = search.get("symbols", {}).get(symbol, {}).get("selection", {})
        state = str(selection.get("state", "NO_RESEARCH_EVIDENCE"))
        blockers = [] if state == "UNTOUCHED_GATE_PASS" else [f"RESEARCH_{state}"]
        blockers.append("NO_RECONCILED_DEMO_FORWARD_EVIDENCE")
        symbols.append(
            {
                "symbol": symbol,
                "metrics": {"research_selection": state},
                "blockers": blockers,
            }
        )
    blockers = [
        f"{row['symbol']}:{blocker}"
        for row in symbols for blocker in row["blockers"]
    ]
    return {
        "strategy_id": "liquidity_breakout",
        "display_name": "H4 M15 M3 Liquidity Breakout",
        "strategy_mode": "liquidity_breakout",
        "status": "RESEARCHING",
        "demo_deployable": False,
        "configuration_hash": strategy_configuration_hash(settings, "liquidity_breakout"),
        "backtest_metrics": {
            "trades": portfolio.get("trades", 0),
            "net_profit": portfolio.get("net_profit"),
            "profit_factor": portfolio.get("profit_factor"),
            "max_drawdown_pct": portfolio.get("max_closed_equity_drawdown_pct"),
        },
        "symbols": symbols,
        "blockers": blockers,
    }


def generate_registry(settings: Settings = SETTINGS) -> dict[str, Any]:
    reconciliation = _reconciliation_proof(settings)
    strategies = [_calibrated_entry(settings, reconciliation), _liquidity_entry(settings)]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status": (
            "DEMO_STRATEGY_READY"
            if any(row["demo_deployable"] for row in strategies)
            else "NO_STRATEGY_READY"
        ),
        "deployment_scope": "DEMO_ONLY",
        "reconciliation": reconciliation,
        "thresholds": {
            "minimum_forward_trades": 30,
            "minimum_forward_sessions": 10,
            "minimum_profit_factor": 1.20,
            "maximum_drawdown_pct": 5.0,
            "positive_expectancy_required": True,
        },
        "strategies": strategies,
    }


def write_dashboard(registry: dict[str, Any], settings: Settings = SETTINGS) -> tuple[Path, Path, Path]:
    settings.strategy_readiness_path.parent.mkdir(parents=True, exist_ok=True)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(registry, indent=2, allow_nan=False) + "\n"
    settings.strategy_readiness_path.write_text(encoded, encoding="utf-8")
    json_path = settings.report_dir / "strategy_readiness.json"
    json_path.write_text(encoded, encoding="utf-8")
    rows = []
    detail_rows = []
    for strategy in registry["strategies"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(strategy['display_name'])}</td>"
            f"<td class='{html.escape(strategy['status'].lower())}'>{html.escape(strategy['status'])}</td>"
            f"<td>{'YES' if strategy['demo_deployable'] else 'NO'}</td>"
            f"<td>{html.escape(', '.join(strategy['blockers']) or 'None')}</td>"
            f"<td><code>{html.escape(strategy['configuration_hash'][:16])}</code></td>"
            "</tr>"
        )
        for symbol in strategy["symbols"]:
            metrics = symbol.get("metrics", {})
            detail_rows.append(
                "<tr>"
                f"<td>{html.escape(strategy['display_name'])}</td>"
                f"<td>{html.escape(symbol['symbol'])}</td>"
                f"<td>{html.escape(str(metrics.get('trades', '—')))}</td>"
                f"<td>{html.escape(str(metrics.get('sessions', '—')))}</td>"
                f"<td>{html.escape(str(metrics.get('net_profit_after_costs', '—')))}</td>"
                f"<td>{html.escape(str(metrics.get('expectancy_after_costs', '—')))}</td>"
                f"<td>{html.escape(str(metrics.get('profit_factor', '—')))}</td>"
                f"<td>{html.escape(str(metrics.get('max_drawdown_pct', '—')))}</td>"
                f"<td>{html.escape(str(metrics.get('research_selection', '—')))}</td>"
                f"<td>{html.escape(', '.join(symbol['blockers']) or 'PASSED')}</td>"
                "</tr>"
            )
    reconciliation = registry["reconciliation"]
    dashboard = f"""<!doctype html><meta charset='utf-8'><title>CryptoAgent Strategy Readiness</title>
<style>body{{font:14px Segoe UI,Arial;margin:32px;background:#0b1220;color:#e5e7eb}}table{{border-collapse:collapse;width:100%;background:#111a2b}}th,td{{border-bottom:1px solid #334155;padding:10px;text-align:left}}th{{color:#93c5fd}}.demo_ready{{color:#4ade80;font-weight:bold}}.rejected,.revoked{{color:#fb7185;font-weight:bold}}.researching{{color:#facc15;font-weight:bold}}code{{color:#c4b5fd}}</style>
<h1>CryptoAgent Strategy Readiness</h1><p>Overall: <strong>{html.escape(registry['overall_status'])}</strong> · Scope: DEMO only · Generated {html.escape(registry['generated_at'])}</p>
<p>Reconciliation: <strong>{'FRESH DEMO' if reconciliation['fresh_demo_reconciliation'] else 'NOT PROVEN'}</strong> · Server: {html.escape(str(reconciliation.get('server') or '—'))} · Synced: {html.escape(str(reconciliation.get('synced_at') or '—'))} · Expires after {html.escape(str(reconciliation['maximum_age_minutes']))} minutes.</p>
<table><thead><tr><th>Strategy</th><th>Status</th><th>DEMO deployable</th><th>Blockers</th><th>Configuration</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Evidence by strategy and symbol</h2>
<table><thead><tr><th>Strategy</th><th>Symbol</th><th>Trades</th><th>Sessions</th><th>Net P/L</th><th>Expectancy</th><th>Profit factor</th><th>Max DD %</th><th>Research result</th><th>Decision</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table>
<p>Pass thresholds: ≥30 reconciled forward DEMO trades, ≥10 sessions, positive net expectancy after costs, profit factor ≥1.20, maximum drawdown ≤5%, approved policy, and fresh DEMO reconciliation.</p>
<p>A launcher cannot bypass this registry. Routing fails closed unless status is DEMO_READY and the configuration hash matches the current strategy.</p>"""
    html_path = settings.report_dir / "strategy_readiness.html"
    html_path.write_text(dashboard, encoding="utf-8")
    return settings.strategy_readiness_path, json_path, html_path


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StrategyDeploymentBlocked(f"strategy readiness registry is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise StrategyDeploymentBlocked(f"invalid strategy readiness registry: {error}") from error
    return payload


def assert_strategy_deployment_ready(settings: Settings = SETTINGS) -> None:
    if not settings.trading_enabled or settings.dry_run:
        return
    registry = load_registry(settings.strategy_readiness_path)
    generated_at = _parse_utc(registry.get("generated_at"))
    now = datetime.now(timezone.utc)
    if not generated_at or not (
        timedelta(0) <= now - generated_at <= timedelta(minutes=settings.readiness_max_age_minutes)
    ):
        raise StrategyDeploymentBlocked("strategy readiness registry is missing, invalid, or stale")
    reconciliation = registry.get("reconciliation", {})
    if not reconciliation.get("fresh_demo_reconciliation", False):
        raise StrategyDeploymentBlocked("fresh successful DEMO reconciliation is not proven")
    strategy = next(
        (
            row for row in registry.get("strategies", [])
            if row.get("strategy_mode") == settings.strategy_mode
        ),
        None,
    )
    if strategy is None:
        raise StrategyDeploymentBlocked(
            f"no readiness record exists for strategy mode {settings.strategy_mode}"
        )
    if strategy.get("status") not in VALID_STATUSES:
        raise StrategyDeploymentBlocked("readiness registry contains an unsupported status")
    if strategy.get("status") != DEMO_READY or not strategy.get("demo_deployable", False):
        blockers = ", ".join(strategy.get("blockers", [])) or "UNSPECIFIED_BLOCKER"
        raise StrategyDeploymentBlocked(
            f"{settings.strategy_mode} is not DEMO_READY: {blockers}"
        )
    current_hash = strategy_configuration_hash(settings, settings.strategy_mode)
    if strategy.get("configuration_hash") != current_hash:
        raise StrategyDeploymentBlocked(
            "strategy configuration changed after readiness approval; regenerate evidence"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate")
    subparsers.add_parser("status")
    check = subparsers.add_parser("check")
    check.add_argument("--strategy-mode", choices=("calibrated", "liquidity_breakout"))
    args = parser.parse_args()
    settings = SETTINGS
    if args.command == "generate":
        registry = generate_registry(settings)
        paths = write_dashboard(registry, settings)
        print(json.dumps(registry, indent=2))
        for path in paths:
            print(path)
    elif args.command == "status":
        print(json.dumps(load_registry(settings.strategy_readiness_path), indent=2))
    else:
        if args.strategy_mode:
            from dataclasses import replace
            settings = replace(settings, strategy_mode=args.strategy_mode)
        assert_strategy_deployment_ready(settings)
        print(f"{settings.strategy_mode} is DEMO_READY")


if __name__ == "__main__":
    main()
