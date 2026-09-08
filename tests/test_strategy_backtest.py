import unittest

import numpy as np

from execution_agent import Side
from strategy_backtest import Position, _exit_for_bar


class StrategyBacktestTests(unittest.TestCase):
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
