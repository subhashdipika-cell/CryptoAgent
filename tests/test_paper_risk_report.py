import json
import tempfile
import unittest
from pathlib import Path

from config import Settings
from execution_agent import PaperMinimumLotRiskReport, Side
from paper_risk_report import write_paper_risk_report


class PaperRiskReportTests(unittest.TestCase):
    def test_writes_paper_only_json_and_html(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(report_dir=Path(temporary))
            report = PaperMinimumLotRiskReport(
                symbol="XAUUSD+",
                side=Side.BUY,
                equity=10_000.0,
                risk_cap_fraction=0.01,
                volume=0.01,
                atr=100.0,
                stop_atr_multiple=1.5,
                stop_distance=150.0,
                broker_minimum_stop_distance=0.1,
                risk_budget=100.0,
                minimum_lot_risk=150.0,
                risk_shortfall=50.0,
                minimum_equity=15_000.0,
                equity_shortfall=5_000.0,
                maximum_stop_distance=100.0,
                stop_distance_excess=50.0,
                maximum_atr=100.0 / 1.5,
                atr_excess=100.0 / 3.0,
                fits_risk_cap=False,
            )
            exports = write_paper_risk_report(settings, [report])
            payload = json.loads(exports["json"].read_text(encoding="utf-8"))
            html = exports["html"].read_text(encoding="utf-8")
            self.assertEqual(payload["classification"], "PAPER_ONLY")
            self.assertFalse(payload["live_execution_changed"])
            self.assertEqual(payload["risk_cap_fraction"], 0.01)
            self.assertEqual(payload["rows"][0]["minimum_equity"], 15_000.0)
            self.assertIn("PAPER_ONLY", html)
            self.assertIn("Minimum equity", html)


if __name__ == "__main__":
    unittest.main()
