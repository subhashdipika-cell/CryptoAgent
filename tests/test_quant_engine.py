import unittest

import numpy as np

from quant_engine import close_prices, trend_from_predictions, true_range_atr


class QuantEngineTests(unittest.TestCase):
    def setUp(self):
        self.rates = np.zeros(
            20,
            dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8")],
        )
        self.rates["time"] = np.arange(20)
        self.rates["close"] = np.arange(100.0, 120.0)
        self.rates["open"] = self.rates["close"] - 0.5
        self.rates["high"] = self.rates["close"] + 1.0
        self.rates["low"] = self.rates["close"] - 1.0

    def test_compact_close_array(self):
        closes = close_prices(self.rates, 10)
        self.assertEqual(closes.dtype, np.float32)
        self.assertTrue(closes.flags.c_contiguous)
        self.assertEqual(len(closes), 10)

    def test_atr_is_positive(self):
        self.assertGreater(true_range_atr(self.rates, 14), 0)

    def test_trend_mapping(self):
        result = trend_from_predictions(100.0, np.array([101, 102, 103, 104, 105]))
        self.assertEqual(result.direction, "BULLISH")
        self.assertGreater(result.probability, 0.5)


if __name__ == "__main__":
    unittest.main()

