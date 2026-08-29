import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import inspect

from execution_agent import Side
from config import Settings
from strategy_backtest import (
    Position,
    _bars_completed_by_observation,
    _exit_for_bar,
    _research_side_allowed,
    _research_replay_bounds,
    _research_trend_allowed,
    _round_trip_commission,
    prepare_research_policy,
)
import strategy_backtest


class StrategyBacktestTests(unittest.TestCase):
    def test_replay_excludes_bars_not_completed_by_observation_time(self):
        rates = np.zeros(3, dtype=[("time", "i8"), ("close", "f8")])
        rates["time"] = [900, 1500, 2100]
        observed_at = strategy_backtest.datetime.fromtimestamp(
            2400, tz=strategy_backtest.timezone.utc
        )
        completed = _bars_completed_by_observation(rates, 15, observed_at)
        self.assertEqual(completed["time"].tolist(), [900, 1500])

    def test_replay_observation_time_must_be_timezone_aware(self):
        rates = np.zeros(1, dtype=[("time", "i8")])
        with self.assertRaises(ValueError):
            _bars_completed_by_observation(
                rates, 15, strategy_backtest.datetime.fromtimestamp(2400)
            )

    def test_h1_trend_filter_uses_completed_history_direction(self):
        rising = np.arange(1.0, 61.0)
        falling = rising[::-1]
        self.assertTrue(_research_trend_allowed(Side.BUY, rising, 50))
        self.assertFalse(_research_trend_allowed(Side.SELL, rising, 50))
        self.assertTrue(_research_trend_allowed(Side.SELL, falling, 50))
        self.assertFalse(_research_trend_allowed(Side.BUY, falling, 50))
        self.assertTrue(_research_trend_allowed(Side.BUY, rising, 0))

    def test_development_walk_forward_bounds_are_chronological(self):
        start, end, classification, fold = _research_replay_bounds(
            1000,
            200,
            100,
            {"research_replay_window": {
                "classification": "DEVELOPMENT_WALK_FORWARD",
                "fold_id": "fold_1",
                "start_fraction": 0.35,
                "end_fraction": 0.5,
            }},
        )
        self.assertEqual((start, end), (350, 500))
        self.assertEqual(classification, "DEVELOPMENT_WALK_FORWARD")
        self.assertEqual(fold, "fold_1")

    def test_replay_bounds_reject_untouched_claims(self):
        with self.assertRaises(ValueError):
            _research_replay_bounds(
                1000,
                200,
                100,
                {"research_replay_window": {
                    "classification": "UNTOUCHED_TEST",
                    "start_fraction": 0.8,
                    "end_fraction": 1.0,
                }},
            )

    def test_replay_metric_pass_cannot_claim_promotion_or_untouched_evidence(self):
        source = inspect.getsource(strategy_backtest.append_research_experiment)
        self.assertIn("BASIC_REPLAY_METRICS_PASS_UNTOUCHED_NOT_PROVEN", source)
        self.assertIn('"promotion_eligible": False', source)
        self.assertIn('"UNTOUCHED_TEST_NOT_PROVEN"', source)
        self.assertNotIn('"REPLAY_GATE_PASS" if', source)

    def test_research_side_filter_is_explicit_and_fail_closed(self):
        self.assertTrue(_research_side_allowed(Side.BUY, "BUY"))
        self.assertFalse(_research_side_allowed(Side.SELL, "BUY"))
        self.assertTrue(_research_side_allowed(Side.SELL, "BOTH"))
        with self.assertRaises(ValueError):
            _research_side_allowed(Side.BUY, "LONG")

    def test_research_policy_is_approved_only_in_temporary_dry_run_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.json"
            source.write_text(
                json.dumps({
                    "research_take_profit_atr_multiple": 2.5,
                    "research_max_risk_fraction": 0.005,
                    "policies": [{"symbol": "BTCUSD", "enabled": True, "approved": False}],
                }),
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
            self.assertEqual(settings.take_profit_atr_multiple, 2.5)
            self.assertEqual(settings.max_risk_fraction, 0.005)
            self.assertTrue(copied["research_only_replay"])
            self.assertTrue(copied["policies"][0]["approved"])
            original = json.loads(source.read_text(encoding="utf-8"))
            self.assertFalse(original["policies"][0]["approved"])

    def test_research_take_profit_override_rejects_non_positive_value(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.json"
            source.write_text(
                json.dumps({
                    "research_take_profit_atr_multiple": 0,
                    "policies": [{"symbol": "BTCUSD", "enabled": True, "approved": False}],
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                prepare_research_policy(Settings(mt5_terminal_path="test"), source)

    def test_research_risk_override_cannot_exceed_runtime_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.json"
            source.write_text(
                json.dumps({
                    "research_max_risk_fraction": 0.03,
                    "policies": [{"symbol": "BTCUSD", "enabled": True, "approved": False}],
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                prepare_research_policy(
                    Settings(mt5_terminal_path="test", max_risk_fraction=0.02), source
                )

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
