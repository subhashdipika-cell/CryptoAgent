import json
import tempfile
import unittest
from pathlib import Path

from config import Settings
from policy_admin import approve
from predictive_validation import FoldResult, calibrate_h1_policy
from revalidation_scheduler import RevalidationScheduler


def policy(symbol: str, enabled: bool) -> dict:
    return {
        "symbol": symbol,
        "model_name": "BTC-DirectRidge" if "BTC" in symbol else "XAU-DirectRidge",
        "enabled": enabled,
        "approved": False,
        "confidence_threshold": 0.6,
        "m15_edge_bps": 10.0,
        "h1_edge_bps": 10.0,
        "calibration_trades": 10,
        "holdout_trades": 6,
        "holdout_net_bps": 20.0 if enabled else -20.0,
        "holdout_profit_factor": 1.2 if enabled else 0.5,
    }


class PolicyGovernanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.active = root / "active.json"
        self.candidate = root / "candidate.json"
        self.state = root / "state.json"
        self.active.write_text(json.dumps({"policies": [policy("XAUUSD+", True)]}))
        self.settings = Settings(
            decision_policy_path=self.active,
            candidate_policy_path=self.candidate,
            revalidation_state_path=self.state,
            automatic_revalidation=False,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_failed_candidate_cannot_be_approved(self):
        self.candidate.write_text(json.dumps({"policies": [policy("BTCUSD", False)]}))
        with self.assertRaises(PermissionError):
            approve("BTCUSD", self.settings)

    def test_passing_candidate_requires_explicit_approval(self):
        self.candidate.write_text(json.dumps({"policies": [policy("BTCUSD", True)]}))
        promoted = approve("BTCUSD", self.settings)
        self.assertTrue(promoted["approved"])
        self.assertIn("activated_at", promoted)
        active = json.loads(self.active.read_text())
        btc = next(row for row in active["policies"] if row["symbol"] == "BTCUSD")
        self.assertTrue(btc["approved"])
        self.assertEqual(btc["activated_at"], promoted["activated_at"])
        self.assertEqual(active["approval_audit"][-1]["approved_at"], promoted["activated_at"])

    def test_h1_candidate_passes_only_on_positive_untouched_holdout(self):
        folds = [
            FoldResult(
                origin_time=index,
                predicted_edge_bps=20.0,
                confidence=0.80,
                actual_return_bps=20.0,
                direction_correct=True,
                traded=True,
                net_return_bps=10.0,
            )
            for index in range(60)
        ]
        candidate, diagnostics = calibrate_h1_policy("BTCUSD", folds)
        self.assertTrue(candidate["enabled"])
        self.assertEqual(candidate["decision_mode"], "H1_ONLY")
        self.assertFalse(candidate["approved"])
        self.assertEqual(diagnostics["deployment"], "DEMO_ELIGIBLE")

    def test_h1_candidate_fails_negative_untouched_holdout(self):
        folds = [
            FoldResult(
                origin_time=index,
                predicted_edge_bps=20.0,
                confidence=0.80,
                actual_return_bps=20.0 if index < 39 else -20.0,
                direction_correct=index < 39,
                traded=True,
                net_return_bps=10.0 if index < 39 else -30.0,
            )
            for index in range(60)
        ]
        candidate, diagnostics = calibrate_h1_policy("BTCUSD", folds)
        self.assertFalse(candidate["enabled"])
        self.assertEqual(diagnostics["deployment"], "SHADOW_ONLY")

    def test_scheduler_recovers_new_bars_across_restart(self):
        scheduler = RevalidationScheduler(self.settings)
        scheduler.observe({"BTCUSD": tuple(range(100, 600))})
        self.assertEqual(scheduler.state["symbols"]["BTCUSD"]["new_completed_bars"], 0)
        restarted = RevalidationScheduler(self.settings)
        restarted.observe({"BTCUSD": tuple(range(500, 1000))})
        self.assertEqual(restarted.state["symbols"]["BTCUSD"]["new_completed_bars"], 400)


if __name__ == "__main__":
    unittest.main()
