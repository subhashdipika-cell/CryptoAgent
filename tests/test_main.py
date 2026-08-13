import unittest
import json
import tempfile
from pathlib import Path

from decision_engine import CalibratedDecisionEngine
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

    def _engine(self, enabled: bool = True) -> CalibratedDecisionEngine:
        temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(
            {
                "policies": [{
                    "symbol": "BTCUSD", "model_name": "BTC-DirectRidge",
                    "enabled": enabled, "approved": enabled, "confidence_threshold": 0.60,
                    "m15_edge_bps": 10.0, "h1_edge_bps": 10.0,
                    "calibration_trades": 10, "holdout_trades": 6,
                    "holdout_net_bps": 12.0, "holdout_profit_factor": 1.2,
                }]
            },
            temporary,
        )
        temporary.close()
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return CalibratedDecisionEngine(Path(temporary.name))

    def test_degraded_sentiment_is_omitted_and_validated_signal_buys(self):
        m15 = forecast("BULLISH", 0.64)
        h1 = forecast("BULLISH", 0.62)
        m15 = ForecastResult(m15.predictions, m15.direction, m15.probability, 0.01, "BTC-DirectRidge", 15.0)
        h1 = ForecastResult(h1.predictions, h1.direction, h1.probability, 0.01, "BTC-DirectRidge", 18.0)
        result = self._engine().evaluate("BTCUSD", m15, h1, 0.5, True)
        self.assertIs(result.side, Side.BUY)
        self.assertEqual(result.reason, "ENTRY_SIGNAL")
        self.assertAlmostEqual(result.score, 0.63)

    def test_disabled_policy_reports_unvalidated_model(self):
        result = self._engine(False).evaluate(
            "BTCUSD", forecast("BULLISH", 0.9), forecast("BULLISH", 0.9), 0.9, False
        )
        self.assertIsNone(result.side)
        self.assertEqual(result.reason, "UNVALIDATED_MODEL")


if __name__ == "__main__":
    unittest.main()
