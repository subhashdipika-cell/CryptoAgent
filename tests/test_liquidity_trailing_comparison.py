import unittest

import numpy as np

from execution_agent import Side
from liquidity_backtest import ReplayPosition
from liquidity_trailing_comparison import VARIANTS, fetch_closed_paginated, make_position_event


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


class FakeHistoryMT5:
    def symbol_select(self, symbol, selected):
        return True

    def copy_rates_from_pos(self, symbol, timeframe, offset, count):
        values = np.zeros(count, dtype=[("time", "i8")])
        values["time"] = np.arange(offset, offset + count)
        return values

    def last_error(self):
        return (1, "Success")


class LiquidityTrailingComparisonTests(unittest.TestCase):
    def test_paginated_history_is_chronological_and_unique(self):
        result = fetch_closed_paginated(FakeHistoryMT5(), "BTCUSD", 3, 50_005)
        self.assertEqual(len(result), 50_005)
        self.assertTrue(np.all(np.diff(result["time"]) > 0))

    def test_one_r_variant_arms_breakeven_for_next_bar(self):
        position = ReplayPosition("BTCUSD", Side.BUY, 0.01, 0, 100, 98, 98, 106, 2)
        event = make_position_event(VARIANTS[1])
        self.assertIsNone(event(position, bar(100, 102.2, 99, 101.5), 0.01, 0))
        self.assertEqual(position.stop, 100)
        exit_price, reason = event(position, bar(101, 101.5, 99.5, 100), 0.01, 10)
        self.assertEqual(reason, "BREAKEVEN")
        self.assertAlmostEqual(exit_price, 99.9)

    def test_staged_variant_reduces_risk_after_one_r(self):
        position = ReplayPosition("XAUUSD+", Side.BUY, 0.01, 0, 100, 98, 98, 106, 2)
        event = make_position_event(VARIANTS[2])
        self.assertIsNone(event(position, bar(100, 102.1, 99, 101.5), 0.01, 0))
        self.assertEqual(position.stop, 99.5)
        _, reason = event(position, bar(101, 101.2, 99.4, 100), 0.01, 0)
        self.assertEqual(reason, "REDUCED_LOSS")

    def test_staged_variant_trails_one_r_behind_best_excursion(self):
        position = ReplayPosition("BTCUSD", Side.SELL, 0.01, 0, 100, 102, 102, 94, 2)
        event = make_position_event(VARIANTS[2])
        self.assertIsNone(event(position, bar(100, 100.5, 95.5, 96, spread=0), 0.01, 0))
        self.assertAlmostEqual(position.stop, 97.5)


if __name__ == "__main__":
    unittest.main()
