import unittest

from execution_agent import Side
from main import combined_side
from quant_engine import ForecastResult


def forecast(direction: str, probability: float) -> ForecastResult:
    import numpy as np

    return ForecastResult(np.ones(5), direction, probability, 0.01)


class SignalTests(unittest.TestCase):
    def test_timeframe_disagreement_holds(self):
        result = combined_side(forecast("BULLISH", 0.9), forecast("BEARISH", 0.9), 0.9, 0.62)
        self.assertIsNone(result)

    def test_aligned_bullish_signal_buys(self):
        result = combined_side(forecast("BULLISH", 0.8), forecast("BULLISH", 0.8), 0.7, 0.62)
        self.assertIs(result, Side.BUY)


if __name__ == "__main__":
    unittest.main()

