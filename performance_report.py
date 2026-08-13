"""Generate reproducible CSV and HTML reports from the reconciled SQLite journal."""

from __future__ import annotations

import argparse
import csv
import html
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
<h2>Strategy and asset breakdown</h2>
{_table(['Strategy','Symbol','Trades','Win rate %','Net P/L','Profit factor','Max drawdown'], group_rows)}
<h2>Latest completed trades</h2>
{_table(['MT5 exit timestamp','Strategy','Symbol','Side','Volume','Entry','Exit','Slippage','Net P/L','Exit reason'], trade_rows)}
<h2>Decision diagnostics</h2>
{_table(['Symbol','HOLD reason','Cycles'], hold_rows)}
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
