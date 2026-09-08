import unittest

import numpy as np

from quant_engine import close_prices, trend_from_predictions, true_range_atr
from asset_predictive_engine import ASSET_SPECS, DedicatedAssetForecastEngine, supervised_matrix
from config import Settings


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

    def test_asset_models_are_distinct(self):
        self.assertNotEqual(ASSET_SPECS["BTC"].context, ASSET_SPECS["XAU"].context)
        self.assertNotEqual(ASSET_SPECS["BTC"].ridge_alpha, ASSET_SPECS["XAU"].ridge_alpha)

    def test_direct_model_generates_five_predictions_without_future_bar(self):
        rates = np.zeros(
            420,
            dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8")],
        )
        rates["time"] = np.arange(420)
        rates["close"] = 100.0 * np.exp(np.arange(420) * 0.0002 + np.sin(np.arange(420) / 9) * 0.002)
        rates["open"] = rates["close"] * 0.9998
        rates["high"] = rates["close"] * 1.001
        rates["low"] = rates["close"] * 0.999
        engine = DedicatedAssetForecastEngine(Settings())
        before = engine.forecast("BTCUSD", rates[:-1], "15min")
        changed_future = rates.copy()
        changed_future["close"][-1] *= 10
        unchanged = engine.forecast("BTCUSD", changed_future[:-1], "15min")
        np.testing.assert_allclose(before.predictions, unchanged.predictions)
        self.assertEqual(len(before.predictions), 5)
        self.assertEqual(before.model_name, "BTC-DirectRidge")

    def test_supervised_matrix_rejects_short_history(self):
        with self.assertRaises(ValueError):
            supervised_matrix(self.rates, ASSET_SPECS["BTC"])


if __name__ == "__main__":
    unittest.main()
