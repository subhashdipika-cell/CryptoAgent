"""Generate reproducible CSV and HTML reports from the reconciled SQLite journal."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config import SETTINGS, Settings
from execution_agent import MT5ExecutionAgent
from trade_journal import TradeJournal


EXIT_REASONS = {
    0: "CLIENT",
    1: "MOBILE",
    2: "WEB",
    3: "EXPERT",
    4: "STOP_LOSS",
    5: "TAKE_PROFIT",
    6: "STOP_OUT",
}

DEMO_TRADE_MODE = 0
MIN_FORWARD_EVIDENCE_TRADES = 30
INSUFFICIENT_FORWARD_EVIDENCE = "INSUFFICIENT_FORWARD_EVIDENCE"
FORWARD_EVIDENCE_AVAILABLE = "FORWARD_EVIDENCE_AVAILABLE"


@dataclass(frozen=True, slots=True)
class CompletedTrade:
    position_id: int
    account_login: int
    server: str
    symbol: str
    strategy: str
    expert_id: int
    side: str
    entry_time: str
    exit_time: str
    entry_volume: float
    requested_price: float | None
    entry_price: float
    exit_price: float
    slippage: float | None
    stop_loss: float | None
    take_profit: float | None
    atr: float | None
    estimated_risk: float | None
    gross_profit: float
    commission: float
    swap: float
    fee: float
    net_profit: float
    duration_minutes: float
    exit_reason: str
    comments: str


def _iso_time(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat(timespec="seconds")


def _weighted_price(rows: list[sqlite3.Row]) -> float:
    volume = sum(float(row["volume"] or 0.0) for row in rows)
    if volume <= 0:
        return 0.0
    return sum(float(row["price"] or 0.0) * float(row["volume"] or 0.0) for row in rows) / volume


def _strategy(rows: list[sqlite3.Row], settings: Settings) -> str:
    for row in rows:
        comment = str(row["comment"] or "")
        if comment.startswith(f"{settings.application_name}|"):
            return comment.split("|", 1)[1]
    magic = int(rows[0]["expert_id"] or 0)
    return settings.strategy_name if magic == settings.magic_number else "LegacyCryptoAgent"


def completed_trades(
    rows: Iterable[sqlite3.Row],
    settings: Settings = SETTINGS,
    submissions: Iterable[sqlite3.Row] = (),
    orders: Iterable[sqlite3.Row] = (),
) -> list[CompletedTrade]:
    submission_by_order = {
        int(row["order_ticket"]): row for row in submissions if row["order_ticket"] is not None
    }
    order_by_ticket = {int(row["ticket"]): row for row in orders}
    grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[int(row["position_id"])].append(row)
    trades: list[CompletedTrade] = []
    for position_id, position_rows in grouped.items():
        ordered = sorted(position_rows, key=lambda row: (int(row["time_msc"]), int(row["ticket"])))
        entries = [row for row in ordered if int(row["entry"] or 0) == 0]
        exits = [row for row in ordered if int(row["entry"] or 0) in {1, 2, 3}]
        if not entries or not exits:
            continue
        entry_time = int(entries[0]["time_msc"])
        exit_time = int(exits[-1]["time_msc"])
        gross = sum(float(row["profit"] or 0.0) for row in ordered)
        commission = sum(float(row["commission"] or 0.0) for row in ordered)
        swap = sum(float(row["swap"] or 0.0) for row in ordered)
        fee = sum(float(row["fee"] or 0.0) for row in ordered)
        comments = " | ".join(dict.fromkeys(str(row["comment"] or "") for row in ordered if row["comment"]))
        exit_reason_number = int(exits[-1]["reason"] or 0)
        entry_order_ticket = int(entries[0]["order_ticket"] or 0)
        submission = submission_by_order.get(entry_order_ticket)
        order = order_by_ticket.get(entry_order_ticket)
        requested_price = float(submission["requested_price"]) if submission else None
        entry_price = _weighted_price(entries)
        is_buy = int(entries[0]["type"] or 0) == 0
        slippage = None
        if requested_price is not None:
            slippage = entry_price - requested_price if is_buy else requested_price - entry_price
        stop_loss = (
            float(submission["stop_loss"])
            if submission
            else (float(order["stop_loss"]) if order else None)
        )
        take_profit = (
            float(submission["take_profit"])
            if submission
            else (float(order["take_profit"]) if order else None)
        )
        trades.append(
            CompletedTrade(
                position_id=position_id,
                account_login=int(ordered[0]["account_login"]),
                server=str(ordered[0]["server"]),
                symbol=str(entries[0]["symbol"]),
                strategy=_strategy(ordered, settings),
                expert_id=int(entries[0]["expert_id"] or 0),
                side="BUY" if is_buy else "SELL",
                entry_time=_iso_time(entry_time),
                exit_time=_iso_time(exit_time),
                entry_volume=sum(float(row["volume"] or 0.0) for row in entries),
                requested_price=requested_price,
                entry_price=entry_price,
                exit_price=_weighted_price(exits),
                slippage=slippage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=float(submission["atr"]) if submission else None,
                estimated_risk=float(submission["estimated_risk"]) if submission else None,
                gross_profit=gross,
                commission=commission,
                swap=swap,
                fee=fee,
                net_profit=gross + commission + swap + fee,
                duration_minutes=max(0.0, (exit_time - entry_time) / 60_000),
                exit_reason=EXIT_REASONS.get(exit_reason_number, f"REASON_{exit_reason_number}"),
                comments=comments,
            )
        )
    return sorted(trades, key=lambda trade: (trade.exit_time, trade.position_id))


def metrics(trades: list[CompletedTrade]) -> dict[str, Any]:
    results = [trade.net_profit for trade in trades]
    winners = [value for value in results if value > 0]
    losers = [value for value in results if value < 0]
    equity_curve = []
    running = peak = max_drawdown = 0.0
    for value in results:
        running += value
        peak = max(peak, running)
        max_drawdown = min(max_drawdown, running - peak)
        equity_curve.append(running)
    gross_profit = sum(winners)
    gross_loss = sum(losers)
    return {
        "trades": len(trades),
        "wins": len(winners),
        "losses": len(losers),
        "breakeven": len(results) - len(winners) - len(losers),
        "win_rate_pct": 100.0 * len(winners) / len(trades) if trades else 0.0,
        "net_profit": sum(results),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss else (math.inf if gross_profit else 0.0),
        "expectancy": sum(results) / len(trades) if trades else 0.0,
        "average_win": gross_profit / len(winners) if winners else 0.0,
        "average_loss": gross_loss / len(losers) if losers else 0.0,
        "max_closed_trade_drawdown": max_drawdown,
        "costs": -sum(trade.commission + trade.swap + trade.fee for trade in trades),
    }


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _audit_time(row: dict[str, Any]) -> datetime | None:
    for key, value in row.items():
        if key.endswith("_at") and isinstance(value, str):
            return _parse_time(value)
    return None


def policy_activation_windows(
    policy_payload: dict[str, Any],
) -> dict[str, tuple[datetime, datetime | None]]:
    """Return auditable policy-active windows without consulting report results."""
    windows: dict[str, tuple[datetime, datetime | None]] = {}
    for policy in policy_payload.get("policies", []):
        activated_at = policy.get("activated_at")
        if activated_at:
            windows[str(policy["symbol"])] = (_parse_time(str(activated_at)), None)
    audit_rows = []
    for row in policy_payload.get("approval_audit", []):
        occurred_at = _audit_time(row)
        if occurred_at is not None:
            audit_rows.append((occurred_at, row))
    for occurred_at, row in sorted(audit_rows, key=lambda item: item[0]):
        symbol = str(row.get("symbol", ""))
        action = str(row.get("action", "")).upper()
        if not symbol:
            continue
        if action == "MANUAL_APPROVAL":
            windows[symbol] = (occurred_at, None)
        elif "REJECTION" in action and symbol in windows:
            activated_at, _ = windows[symbol]
            if occurred_at >= activated_at:
                windows[symbol] = (activated_at, occurred_at)
    return windows


def demo_accounts(snapshot_rows: Iterable[sqlite3.Row]) -> set[tuple[int, str]]:
    """Use the latest snapshot for each account/server as its reconciled mode proof."""
    latest: dict[tuple[int, str], sqlite3.Row] = {}
    for row in snapshot_rows:
        key = (int(row["account_login"]), str(row["server"]))
        previous = latest.get(key)
        if previous is None or str(row["recorded_at"]) > str(previous["recorded_at"]):
            latest[key] = row
    return {
        key
        for key, row in latest.items()
        if row["trade_mode"] is not None and int(row["trade_mode"]) == DEMO_TRADE_MODE
    }


def forward_evidence(
    trades: list[CompletedTrade],
    snapshot_rows: Iterable[sqlite3.Row],
    policy_payload: dict[str, Any],
    minimum_trades: int = MIN_FORWARD_EVIDENCE_TRADES,
    expert_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Summarize only completed, reconciled DEMO trades entered while policy was active."""
    windows = policy_activation_windows(policy_payload)
    demo = demo_accounts(snapshot_rows)
    symbols = sorted({str(row["symbol"]) for row in policy_payload.get("policies", [])})
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        window = windows.get(symbol)
        eligible: list[CompletedTrade] = []
        if window is not None:
            activated_at, deactivated_at = window
            for trade in trades:
                entry_at = _parse_time(trade.entry_time)
                if (
                    trade.symbol == symbol
                    and (expert_ids is None or trade.expert_id in expert_ids)
                    and (trade.account_login, trade.server) in demo
                    and entry_at >= activated_at
                    and (deactivated_at is None or entry_at < deactivated_at)
                ):
                    eligible.append(trade)
        result = metrics(eligible)
        rows.append(
            {
                "symbol": symbol,
                "evidence_state": (
                    FORWARD_EVIDENCE_AVAILABLE
                    if result["trades"] >= minimum_trades
                    else INSUFFICIENT_FORWARD_EVIDENCE
                ),
                "minimum_sample_size": minimum_trades,
                "sample_size": result["trades"],
                "activation_at": window[0].isoformat(timespec="seconds") if window else None,
                "deactivation_at": (
                    window[1].isoformat(timespec="seconds")
                    if window and window[1] is not None
                    else None
                ),
                "net_profit_after_costs": result["net_profit"],
                "costs": result["costs"],
                "max_drawdown": abs(result["max_closed_trade_drawdown"]),
                "win_rate_pct": result["win_rate_pct"],
                "profit_factor": result["profit_factor"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "∞"
        return f"{value:.2f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_format(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def generate_report(settings: Settings = SETTINGS) -> dict[str, Path]:
    journal = TradeJournal(settings)
    deal_rows = journal.rows("mt5_deals")
    submission_rows = journal.rows("submissions")
    order_rows = journal.rows("mt5_orders")
    trades = completed_trades(deal_rows, settings, submission_rows, order_rows)
    output = Path(settings.report_dir)
    output.mkdir(parents=True, exist_ok=True)

    exports: dict[str, Path] = {}
    for table, filename in (
        ("mt5_deals", "deals.csv"),
        ("mt5_orders", "orders.csv"),
        ("signals", "signals.csv"),
        ("model_forecasts", "model_forecasts.csv"),
        ("account_snapshots", "equity_snapshots.csv"),
        ("submissions", "submissions.csv"),
        ("order_plan_rejections", "order_plan_rejections.csv"),
        ("liquidity_signals", "liquidity_signals.csv"),
    ):
        path = output / filename
        _write_csv(path, [dict(row) for row in journal.rows(table)])
        exports[table] = path
    trades_path = output / "completed_trades.csv"
    _write_csv(trades_path, [asdict(trade) for trade in trades])
    exports["completed_trades"] = trades_path

    overall = metrics(trades)
    accounts = sorted({f"{trade.server} / {trade.account_login}" for trade in trades})
    snapshots = journal.rows("account_snapshots")
    policy_path = Path(settings.decision_policy_path)
    policy_payload = (
        json.loads(policy_path.read_text(encoding="utf-8"))
        if policy_path.is_file()
        else {"policies": []}
    )
    evidence_rows = forward_evidence(
        trades,
        snapshots,
        policy_payload,
        expert_ids={settings.magic_number},
    )
    evidence_path = output / "forward_evidence.csv"
    _write_csv(evidence_path, evidence_rows)
    exports["forward_evidence"] = evidence_path
    signal_rows = journal.rows("signals")
    latest_equity = float(snapshots[-1]["equity"]) if snapshots else None
    groups: dict[tuple[str, str], list[CompletedTrade]] = defaultdict(list)
    for trade in trades:
        groups[(trade.strategy, trade.symbol)].append(trade)
    group_rows = []
    for (strategy, symbol), group in sorted(groups.items()):
        result = metrics(group)
        group_rows.append(
            [strategy, symbol, result["trades"], result["win_rate_pct"], result["net_profit"], result["profit_factor"], result["max_closed_trade_drawdown"]]
        )
    trade_rows = [
        [trade.exit_time, trade.strategy, trade.symbol, trade.side, trade.entry_volume, trade.entry_price, trade.exit_price, trade.slippage, trade.net_profit, trade.exit_reason]
        for trade in reversed(trades[-100:])
    ]
    hold_reasons: dict[tuple[str, str], int] = defaultdict(int)
    for row in signal_rows:
        if row["decision"] == "HOLD":
            hold_reasons[(str(row["symbol"]), str(row["decision_reason"]))] += 1
    hold_rows = [
        [symbol, reason, count]
        for (symbol, reason), count in sorted(hold_reasons.items())
    ]
    rejection_rows = [
        [
            row["recorded_at"], row["symbol"], row["side"], row["rejection_error"],
            row["risk_shortfall"], row["equity_shortfall"], row["minimum_equity"],
            row["maximum_stop_distance"], row["maximum_atr"],
        ]
        for row in reversed(journal.rows("order_plan_rejections")[-100:])
    ]
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CryptoAgent Performance Report</title>
<style>
body{{font:14px system-ui;margin:32px;background:#0b1220;color:#e5e7eb}}h1,h2{{color:#f8fafc}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.card{{background:#172033;padding:16px;border-radius:10px}}.value{{font-size:24px;font-weight:700}}
table{{border-collapse:collapse;width:100%;background:#111a2b;margin:12px 0 28px}}th,td{{padding:9px;border-bottom:1px solid #334155;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{color:#93c5fd}}.note{{color:#94a3b8}}
</style></head><body>
<h1>CryptoAgent Performance Report</h1>
<p class="note">Generated {html.escape(generated_at)} from reconciled MT5 deals for {html.escape(', '.join(accounts) if accounts else 'no reconciled account yet')}. Trade timestamps reflect MT5 history time. Open positions and unfilled orders are excluded from realized metrics.</p>
{f'<p class="note">Latest recorded equity: {latest_equity:.2f}</p>' if latest_equity is not None else ''}
<div class="grid">{''.join(f'<div class="card"><div>{html.escape(key.replace("_", " ").title())}</div><div class="value">{html.escape(_format(value))}</div></div>' for key, value in overall.items())}</div>
<h2>Forward evidence after policy activation</h2>
<p class="note">Read-only evidence from fully reconciled deals whose entry occurred while the asset policy was active and whose account/server latest snapshot is MT5 DEMO. A minimum of {MIN_FORWARD_EVIDENCE_TRADES} completed trades per asset is required to leave {INSUFFICIENT_FORWARD_EVIDENCE}. These results do not alter policy, eligibility, sizing, or routing.</p>
{_table(['Symbol','Evidence state','Minimum sample','Sample size','Activated at (UTC)','Deactivated at (UTC)','Net P/L after costs','Costs','Max drawdown','Win rate %','Profit factor'], [[row['symbol'], row['evidence_state'], row['minimum_sample_size'], row['sample_size'], row['activation_at'], row['deactivation_at'], row['net_profit_after_costs'], row['costs'], row['max_drawdown'], row['win_rate_pct'], row['profit_factor']] for row in evidence_rows])}
<h2>Strategy and asset breakdown</h2>
{_table(['Strategy','Symbol','Trades','Win rate %','Net P/L','Profit factor','Max drawdown'], group_rows)}
<h2>Latest completed trades</h2>
{_table(['MT5 exit timestamp','Strategy','Symbol','Side','Volume','Entry','Exit','Slippage','Net P/L','Exit reason'], trade_rows)}
<h2>Decision diagnostics</h2>
{_table(['Symbol','HOLD reason','Cycles'], hold_rows)}
<h2>ORDER_PLAN_REJECTED paper diagnostics</h2>
<p class="note">These diagnostics use a fixed 1% paper reference for 0.01 lot and do not change runtime sizing.</p>
{_table(['Recorded at','Symbol','Side','Rejection','Risk shortfall','Equity shortfall','Minimum equity','Maximum stop','Maximum ATR'], rejection_rows)}
<p class="note">Attribution uses Expert IDs {html.escape(str(settings.tracked_magic_numbers))}; broker comments are descriptive and may be overwritten on SL/TP exits.</p>
</body></html>"""
    report_path = output / "performance_report.html"
    report_path.write_text(report, encoding="utf-8")
    exports["html"] = report_path
    return exports


def sync_from_terminal(settings: Settings = SETTINGS) -> dict[str, int]:
    journal = TradeJournal(settings)
    agent = MT5ExecutionAgent(settings)
    agent.connect()
    try:
        account = agent.mt5.account_info()
        if account is None:
            raise RuntimeError("MT5 account unavailable")
        journal.record_account(account, agent.snapshot())
        return journal.sync_mt5_history(agent.mt5, account)
    finally:
        agent.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true", help="reconcile MT5 history before reporting")
    args = parser.parse_args()
    SETTINGS.validate()
    if args.sync:
        counts = sync_from_terminal()
        print(f"Reconciled {counts['orders']} orders and {counts['deals']} deals")
    exports = generate_report()
    print(f"HTML report: {exports['html']}")
    print(f"Completed trades CSV: {exports['completed_trades']}")


if __name__ == "__main__":
    main()
