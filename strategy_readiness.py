"""Auditable strategy readiness registry and fail-closed DEMO routing gate."""

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
DEMO_ONLY = "DEMO_ONLY"
VALID_STATUSES = {"RESEARCHING", "REJECTED", "REVOKED", DEMO_READY}


class StrategyDeploymentBlocked(PermissionError):
    """Raised before engine initialization when routing evidence is not valid."""


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def strategy_configuration_hash(settings: Settings) -> str:
    """Bind readiness to every routing-relevant policy, engine, and risk input."""
    if not settings.decision_policy_path.is_file():
        raise StrategyDeploymentBlocked(
            f"decision policy is missing: {settings.decision_policy_path}"
        )
    digest = hashlib.sha256()
    for path in (
        settings.decision_policy_path,
        BASE_DIR / "decision_engine.py",
        BASE_DIR / "asset_predictive_engine.py",
        BASE_DIR / "execution_agent.py",
        BASE_DIR / "main.py",
        BASE_DIR / "strategy_readiness.py",
    ):
        if not path.is_file():
            raise StrategyDeploymentBlocked(f"strategy component is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(
            {
                "symbols": settings.symbols,
                "predictive_mode": settings.predictive_mode,
                "strategy_name": settings.strategy_name,
                "magic_number": settings.magic_number,
                "max_risk_fraction": settings.max_risk_fraction,
                "max_margin_fraction": settings.max_margin_fraction,
                "stop_atr_multiple": settings.stop_atr_multiple,
                "take_profit_atr_multiple": settings.take_profit_atr_multiple,
                "trailing_trigger_atr": settings.trailing_trigger_atr,
                "trailing_distance_atr": settings.trailing_distance_atr,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _fresh_reconciliation(settings: Settings) -> dict[str, Any]:
    marker = _load_json(settings.report_dir / "reconciliation_status.json", {})
    synced_at = _parse_utc(marker.get("synced_at"))
    now = datetime.now(timezone.utc)
    fresh = bool(
        synced_at
        and timedelta(0) <= now - synced_at
        <= timedelta(minutes=settings.readiness_max_age_minutes)
    )
    proven = bool(
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
        "fresh_demo_reconciliation": proven,
    }


def generate_registry(settings: Settings = SETTINGS) -> dict[str, Any]:
    policy_payload = _load_json(
        settings.decision_policy_path, {"policies": [], "approval_audit": []}
    )
    policies = {
        str(row.get("symbol")): row for row in policy_payload.get("policies", [])
    }
    evidence = {
        str(row.get("symbol")): row
        for row in _load_csv(settings.report_dir / "forward_evidence.csv")
    }
    reconciliation = _fresh_reconciliation(settings)
    rejected_symbols = {
        str(row.get("symbol"))
        for row in policy_payload.get("approval_audit", [])
        if "REJECTION" in str(row.get("action", "")).upper()
    }
    symbol_rows: list[dict[str, Any]] = []
    for symbol in settings.symbols:
        policy = policies.get(symbol, {})
        row = evidence.get(symbol, {})
        trades = int(row.get("sample_size") or 0)
        sessions = int(row.get("sessions") or 0)
        expectancy = _number(row, "expectancy_after_costs")
        profit_factor = _number(row, "profit_factor")
        drawdown_pct = _number(row, "max_drawdown_pct")
        blockers: list[str] = []
        if not reconciliation["fresh_demo_reconciliation"]:
            blockers.append("FRESH_DEMO_RECONCILIATION_REQUIRED")
        if not policy.get("enabled", False):
            blockers.append("POLICY_DISABLED")
        if not policy.get("approved", False):
            blockers.append("POLICY_NOT_APPROVED")
        if trades < 30:
            blockers.append(f"FORWARD_TRADES_{trades}_OF_30")
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
        symbol_rows.append(
            {
                "symbol": symbol,
                "status": (
                    DEMO_READY
                    if not blockers
                    else ("REVOKED" if symbol in rejected_symbols else "REJECTED")
                ),
                "demo_deployable": not blockers,
                "metrics": {
                    "trades": trades,
                    "sessions": sessions,
                    "net_profit_after_costs": _number(row, "net_profit_after_costs"),
                    "expectancy_after_costs": expectancy,
                    "profit_factor": profit_factor,
                    "max_drawdown_pct": drawdown_pct,
                },
                "blockers": blockers,
            }
        )
    all_ready = bool(symbol_rows) and all(row["demo_deployable"] for row in symbol_rows)
    blockers = [
        f"{row['symbol']}:{blocker}"
        for row in symbol_rows
        for blocker in row["blockers"]
    ]
    strategy = {
        "strategy_id": "asset_calibrated",
        "display_name": "BTC and Gold Asset Calibrated",
        "strategy_mode": "calibrated",
        "status": DEMO_READY if all_ready else (
            "REVOKED" if rejected_symbols else "REJECTED"
        ),
        "demo_deployable": all_ready,
        "configuration_hash": strategy_configuration_hash(settings),
        "symbols": symbol_rows,
        "blockers": blockers,
    }
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status": "DEMO_STRATEGY_READY" if all_ready else "NO_STRATEGY_READY",
        "deployment_scope": DEMO_ONLY,
        "reconciliation": reconciliation,
        "thresholds": {
            "minimum_forward_trades": 30,
            "minimum_forward_sessions": 10,
            "minimum_profit_factor": 1.20,
            "maximum_drawdown_pct": 5.0,
            "positive_expectancy_required": True,
        },
        "strategies": [strategy],
    }


def write_dashboard(
    registry: dict[str, Any], settings: Settings = SETTINGS
) -> tuple[Path, Path, Path]:
    settings.strategy_readiness_path.parent.mkdir(parents=True, exist_ok=True)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(registry, indent=2, allow_nan=False) + "\n"
    settings.strategy_readiness_path.write_text(encoded, encoding="utf-8")
    report_json = settings.report_dir / "strategy_readiness.json"
    report_json.write_text(encoded, encoding="utf-8")
    rows = []
    for strategy in registry["strategies"]:
        for symbol in strategy["symbols"]:
            metrics = symbol["metrics"]
            rows.append(
                "<tr>"
                f"<td>{html.escape(symbol['symbol'])}</td>"
                f"<td>{html.escape(symbol['status'])}</td>"
                f"<td>{metrics['trades']}</td><td>{metrics['sessions']}</td>"
                f"<td>{html.escape(str(metrics['expectancy_after_costs']))}</td>"
                f"<td>{html.escape(str(metrics['profit_factor']))}</td>"
                f"<td>{html.escape(str(metrics['max_drawdown_pct']))}</td>"
                f"<td>{html.escape(', '.join(symbol['blockers']) or 'PASSED')}</td>"
                "</tr>"
            )
    reconciliation = registry["reconciliation"]
    document = f"""<!doctype html><meta charset='utf-8'><title>CryptoAgent Strategy Readiness</title>
<style>body{{font:14px Segoe UI,Arial;margin:32px;background:#0b1220;color:#e5e7eb}}table{{border-collapse:collapse;width:100%;background:#111a2b}}th,td{{border-bottom:1px solid #334155;padding:10px;text-align:left}}th{{color:#93c5fd}}code{{color:#c4b5fd}}</style>
<h1>CryptoAgent Strategy Readiness</h1>
<p>Overall: <strong>{html.escape(registry['overall_status'])}</strong> · Scope: DEMO only · Generated: {html.escape(registry['generated_at'])}</p>
<p>Reconciliation: <strong>{'FRESH DEMO' if reconciliation['fresh_demo_reconciliation'] else 'NOT PROVEN'}</strong> · Server: {html.escape(str(reconciliation.get('server') or '—'))} · Expires after {reconciliation['maximum_age_minutes']} minutes.</p>
<table><thead><tr><th>Symbol</th><th>Status</th><th>Trades</th><th>Sessions</th><th>Expectancy</th><th>Profit factor</th><th>Max DD %</th><th>Blockers</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Every configured symbol must pass: 30 reconciled DEMO trades, 10 sessions, positive cost-adjusted expectancy, PF ≥1.20, DD ≤5%, approved policy, fresh reconciliation, and an unchanged configuration hash.</p>
<p>Backtests are research evidence only. Readiness permits DEMO routing only and never guarantees profit.</p>"""
    report_html = settings.report_dir / "strategy_readiness.html"
    report_html.write_text(document, encoding="utf-8")
    return settings.strategy_readiness_path, report_json, report_html


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StrategyDeploymentBlocked(f"strategy readiness registry is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise StrategyDeploymentBlocked(f"invalid strategy readiness registry: {error}") from error


def assert_strategy_deployment_ready(settings: Settings = SETTINGS) -> None:
    if not settings.trading_enabled or settings.dry_run:
        return
    if not settings.require_demo_account:
        raise StrategyDeploymentBlocked("autonomous routing is restricted to DEMO accounts")
    registry = load_registry(settings.strategy_readiness_path)
    generated_at = _parse_utc(registry.get("generated_at"))
    now = datetime.now(timezone.utc)
    if not generated_at or not (
        timedelta(0) <= now - generated_at
        <= timedelta(minutes=settings.readiness_max_age_minutes)
    ):
        raise StrategyDeploymentBlocked("strategy readiness registry is missing, invalid, or stale")
    if registry.get("deployment_scope") != DEMO_ONLY:
        raise StrategyDeploymentBlocked("strategy readiness scope is not DEMO_ONLY")
    if not registry.get("reconciliation", {}).get("fresh_demo_reconciliation", False):
        raise StrategyDeploymentBlocked("fresh successful DEMO reconciliation is not proven")
    strategy = next(
        (
            row for row in registry.get("strategies", [])
            if row.get("strategy_mode") == "calibrated"
        ),
        None,
    )
    if strategy is None:
        raise StrategyDeploymentBlocked("calibrated strategy readiness record is missing")
    if strategy.get("status") not in VALID_STATUSES:
        raise StrategyDeploymentBlocked("readiness registry contains an unsupported status")
    if strategy.get("status") != DEMO_READY or not strategy.get("demo_deployable", False):
        blockers = ", ".join(strategy.get("blockers", [])) or "UNSPECIFIED_BLOCKER"
        raise StrategyDeploymentBlocked(f"calibrated strategy is not DEMO_READY: {blockers}")
    if strategy.get("configuration_hash") != strategy_configuration_hash(settings):
        raise StrategyDeploymentBlocked(
            "strategy configuration changed after readiness generation"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "status", "check"))
    args = parser.parse_args()
    if args.command == "generate":
        registry = generate_registry()
        paths = write_dashboard(registry)
        print(json.dumps(registry, indent=2))
        for path in paths:
            print(path)
    elif args.command == "status":
        print(json.dumps(load_registry(SETTINGS.strategy_readiness_path), indent=2))
    else:
        assert_strategy_deployment_ready(SETTINGS)
        print("calibrated strategy is DEMO_READY")


if __name__ == "__main__":
    try:
        main()
    except StrategyDeploymentBlocked as error:
        print(f"BLOCKED: {error}")
        raise SystemExit(1) from None
