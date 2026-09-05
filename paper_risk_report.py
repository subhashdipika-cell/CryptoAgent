"""Generate a read-only XAUUSD 0.01-lot report against a fixed 1% risk cap."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from config import SETTINGS, Settings
from execution_agent import MT5ExecutionAgent, PaperMinimumLotRiskReport, Side
from quant_engine import true_range_atr


PAPER_RISK_CAP_FRACTION = 0.01
PAPER_VOLUME = 0.01


def _row(report: PaperMinimumLotRiskReport) -> dict[str, object]:
    row = asdict(report)
    row["side"] = report.side.value
    row["broker_stop_fits_risk_cap"] = (
        report.maximum_stop_distance + 1e-9 >= report.broker_minimum_stop_distance
    )
    return row


def write_paper_risk_report(
    settings: Settings,
    reports: list[PaperMinimumLotRiskReport],
) -> dict[str, Path]:
    """Write auditable JSON and HTML artifacts without submitting or sizing an order."""
    if not reports:
        raise ValueError("at least one paper risk calculation is required")
    output = Path(settings.report_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [_row(report) for report in reports]
    payload = {
        "generated_at": generated_at,
        "classification": "PAPER_ONLY",
        "risk_cap_fraction": PAPER_RISK_CAP_FRACTION,
        "paper_volume": PAPER_VOLUME,
        "live_execution_changed": False,
        "rows": rows,
    }
    json_path = output / "xau_minimum_equity_risk_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['symbol']))}</td>"
        f"<td>{html.escape(str(row['side']))}</td>"
        f"<td>{float(row['equity']):.2f}</td>"
        f"<td>{float(row['risk_budget']):.2f}</td>"
        f"<td>{float(row['minimum_lot_risk']):.2f}</td>"
        f"<td>{float(row['risk_shortfall']):.2f}</td>"
        f"<td>{float(row['minimum_equity']):.2f}</td>"
        f"<td>{float(row['equity_shortfall']):.2f}</td>"
        f"<td>{float(row['atr']):.5f}</td>"
        f"<td>{float(row['maximum_atr']):.5f}</td>"
        f"<td>{float(row['stop_distance']):.5f}</td>"
        f"<td>{float(row['maximum_stop_distance']):.5f}</td>"
        f"<td>{float(row['broker_minimum_stop_distance']):.5f}</td>"
        f"<td>{'YES' if row['fits_risk_cap'] else 'NO'}</td>"
        "</tr>"
        for row in rows
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XAUUSD 0.01-Lot Paper Risk Conditions</title>
<style>body{{font:14px system-ui;margin:32px;background:#0b1220;color:#e5e7eb}}table{{border-collapse:collapse;width:100%;background:#111a2b}}th,td{{padding:8px;border-bottom:1px solid #334155;text-align:right}}th{{color:#93c5fd}}.note{{color:#fbbf24}}</style>
</head><body><h1>XAUUSD 0.01-Lot Paper Risk Conditions</h1>
<p class="note">PAPER_ONLY: fixed 1% reference cap. This report does not alter runtime sizing, submit orders, change live execution, or enable BTC.</p>
<p>Generated {html.escape(generated_at)}. Minimum equity is calculated at the current ATR-derived stop. Maximum ATR and stop distance are the largest values that fit 0.01 lot at the current equity.</p>
<table><thead><tr><th>Symbol</th><th>Side</th><th>Equity</th><th>1% budget</th><th>0.01 risk</th><th>Risk shortfall</th><th>Minimum equity</th><th>Equity shortfall</th><th>Current ATR</th><th>Maximum ATR</th><th>Current stop</th><th>Maximum stop</th><th>Broker minimum stop</th><th>Fits?</th></tr></thead><tbody>{table_rows}</tbody></table>
</body></html>"""
    html_path = output / "xau_minimum_equity_risk_report.html"
    html_path.write_text(document, encoding="utf-8")
    return {"json": json_path, "html": html_path}


def generate_from_terminal(settings: Settings = SETTINGS) -> dict[str, Path]:
    """Read DEMO account/broker state and completed bars; never route an order."""
    paper_settings = replace(
        settings,
        trading_enabled=False,
        dry_run=True,
        require_demo_account=True,
    )
    xau_symbols = [symbol for symbol in paper_settings.symbols if "XAU" in symbol.upper() or "GOLD" in symbol.upper()]
    if len(xau_symbols) != 1:
        raise ValueError("exactly one configured XAUUSD/Gold symbol is required")
    agent = MT5ExecutionAgent(paper_settings)
    agent.connect()
    try:
        symbol = xau_symbols[0]
        rates = agent.bars(symbol, agent.mt5.TIMEFRAME_M15, paper_settings.bar_count)
        atr = true_range_atr(rates, paper_settings.atr_period)
        reports = [
            agent.paper_minimum_lot_risk_report(
                symbol,
                side,
                atr,
                risk_cap_fraction=PAPER_RISK_CAP_FRACTION,
                volume=PAPER_VOLUME,
            )
            for side in (Side.BUY, Side.SELL)
        ]
        return write_paper_risk_report(paper_settings, reports)
    finally:
        agent.shutdown()


def main() -> None:
    SETTINGS.validate()
    exports = generate_from_terminal()
    print(f"PAPER_ONLY HTML report: {exports['html']}")
    print(f"PAPER_ONLY JSON evidence: {exports['json']}")


if __name__ == "__main__":
    main()
