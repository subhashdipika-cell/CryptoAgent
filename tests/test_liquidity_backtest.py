import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from execution_agent import Side
from config import Settings
from liquidity_backtest import ReplayPosition, _closed_slice, _position_event, append_experiment


def bar(open_price, high, low, close, spread=0):
    row = np.zeros(
        (),
        dtype=[
            ("open", "f8"), ("high", "f8"), ("low", "f8"),
            ("close", "f8"), ("spread", "i4"),
        ],
    )
    row["open"], row["high"], row["low"], row["close"], row["spread"] = (
        open_price, high, low, close, spread,
    )
    return row


class LiquidityBacktestTests(unittest.TestCase):
    def test_experiment_ledger_is_research_only_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "liquidity_experiments.jsonl"
            append_experiment(
                ledger,
                Settings(),
                30_000,
                1_000.0,
                3.0,
                10.0,
                [{"symbol": "BTCUSD", "trades": 4, "profit_factor": 1.1}],
            )
            record = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(record["evidence_class"], "HISTORICAL_BACKTEST_NOT_FORWARD_EVIDENCE")
            self.assertFalse(record["promotion_eligible"])
            self.assertFalse(record["routing_changed"])
            self.assertIn("WALK_FORWARD_STABILITY_NOT_PROVEN", record["promotion_blockers"])

    def test_higher_timeframe_slice_excludes_unclosed_bar(self):
        rates = np.zeros(4, dtype=[("time", "i8")])
        rates["time"] = [0, 900, 1800, 2700]
        known = _closed_slice(rates, trigger_open=1800, timeframe_seconds=900, limit=20)
        self.assertEqual(known["time"].tolist(), [0, 900])

    def test_breakeven_is_armed_only_after_trigger_bar_closes(self):
        position = ReplayPosition(
            "BTCUSD", Side.BUY, 0.01, 0, 100.0, 98.0, 98.0, 106.0, 2.0,
        )
        self.assertIsNone(_position_event(position, bar(100, 104.5, 99.0, 103), 0.01, 0))
        self.assertTrue(position.breakeven_armed)
        self.assertEqual(position.stop, 100.0)
        exit_price, reason = _position_event(position, bar(103, 103.5, 99.5, 100), 0.01, 10)
        self.assertEqual(reason, "BREAKEVEN")
        self.assertAlmostEqual(exit_price, 99.9)

    def test_ambiguous_bar_is_conservatively_stop_first(self):
        position = ReplayPosition(
            "XAUUSD+", Side.BUY, 0.01, 0, 100.0, 98.0, 98.0, 105.0, 2.0,
        )
        exit_price, reason = _position_event(position, bar(100, 106, 97, 102), 0.01, 10)
        self.assertEqual(reason, "AMBIGUOUS_STOP_FIRST")
        self.assertAlmostEqual(exit_price, 97.9)

    def test_sell_stop_uses_historical_spread(self):
        position = ReplayPosition(
            "BTCUSD", Side.SELL, 0.01, 0, 100.0, 102.0, 102.0, 94.0, 2.0,
        )
        exit_price, reason = _position_event(
            position, bar(101.8, 101.9, 99.0, 101.0, spread=20), point=0.01, slippage_points=10,
        )
        self.assertEqual(reason, "STOP_LOSS")
        self.assertAlmostEqual(exit_price, 102.1)


if __name__ == "__main__":
    unittest.main()
