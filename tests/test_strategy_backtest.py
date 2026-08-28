import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from execution_agent import Side
from config import Settings
from strategy_backtest import (
    Position,
    _exit_for_bar,
    _round_trip_commission,
    prepare_research_policy,
)


class StrategyBacktestTests(unittest.TestCase):
    def test_research_policy_is_approved_only_in_temporary_dry_run_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.json"
            source.write_text(
                json.dumps({"policies": [{"symbol": "BTCUSD", "enabled": True, "approved": False}]}),
                encoding="utf-8",
            )
            settings, temporary = prepare_research_policy(
                Settings(trading_enabled=True, dry_run=False, mt5_terminal_path="test"),
                source,
            )
            self.addCleanup(temporary.unlink, missing_ok=True)
            copied = json.loads(temporary.read_text(encoding="utf-8"))

            self.assertFalse(settings.trading_enabled)
            self.assertTrue(settings.dry_run)
            self.assertTrue(copied["research_only_replay"])
            self.assertTrue(copied["policies"][0]["approved"])
            original = json.loads(source.read_text(encoding="utf-8"))
            self.assertFalse(original["policies"][0]["approved"])

    def test_commission_scales_with_volume_at_observed_broker_rate(self):
        self.assertAlmostEqual(_round_trip_commission(0.01, 3.0), 0.06)
        self.assertAlmostEqual(_round_trip_commission(0.19, 3.0), 1.14)

    def test_ambiguous_bar_is_resolved_stop_first(self):
        row = np.zeros(
            (),
            dtype=[("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"), ("spread", "i4")],
        )
        row["open"], row["high"], row["low"], row["close"] = 100, 104, 96, 101
        position = Position("XAUUSD+", Side.BUY, 0.01, 0, 100, 98, 103, 1, 2)
        exit_price, reason = _exit_for_bar(position, row, point=0.01, slippage_points=10)
        self.assertEqual(reason, "AMBIGUOUS_STOP_FIRST")
        self.assertAlmostEqual(exit_price, 97.9)


if __name__ == "__main__":
    unittest.main()
