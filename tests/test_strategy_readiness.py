import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from config import Settings
from main import TradingApplication
from strategy_readiness import (
    DEMO_READY,
    StrategyDeploymentBlocked,
    assert_strategy_deployment_ready,
    strategy_configuration_hash,
)


class StrategyReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.policy = root / "policy.json"
        self.registry = root / "readiness.json"
        self.policy.write_text(json.dumps({"policies": []}), encoding="utf-8")
        self.settings = Settings(
            symbols=("BTCUSD", "XAUUSD+"),
            decision_policy_path=self.policy,
            strategy_readiness_path=self.registry,
            strategy_mode="calibrated",
            trading_enabled=True,
            dry_run=False,
            mt5_terminal_path="D:/does-not-need-to-exist-for-this-test.exe",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, status: str, configuration_hash: str) -> None:
        self.registry.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "reconciliation": {"fresh_demo_reconciliation": True},
                    "strategies": [
                        {
                            "strategy_mode": "calibrated",
                            "status": status,
                            "demo_deployable": status == DEMO_READY,
                            "configuration_hash": configuration_hash,
                            "blockers": ["TEST_BLOCKER"] if status != DEMO_READY else [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def test_missing_registry_blocks_routing(self):
        with self.assertRaises(StrategyDeploymentBlocked):
            assert_strategy_deployment_ready(self.settings)

    def test_non_ready_status_blocks_routing(self):
        self._write("REJECTED", strategy_configuration_hash(self.settings, "calibrated"))
        with self.assertRaisesRegex(StrategyDeploymentBlocked, "TEST_BLOCKER"):
            assert_strategy_deployment_ready(self.settings)

    def test_matching_demo_ready_configuration_allows_routing(self):
        self._write(DEMO_READY, strategy_configuration_hash(self.settings, "calibrated"))
        assert_strategy_deployment_ready(self.settings)

    def test_stale_configuration_hash_blocks_routing(self):
        self._write(DEMO_READY, "stale")
        with self.assertRaisesRegex(StrategyDeploymentBlocked, "configuration changed"):
            assert_strategy_deployment_ready(self.settings)

    def test_stale_registry_blocks_routing(self):
        self._write(DEMO_READY, strategy_configuration_hash(self.settings, "calibrated"))
        payload = json.loads(self.registry.read_text(encoding="utf-8"))
        payload["generated_at"] = (
            datetime.now(timezone.utc) - timedelta(minutes=16)
        ).isoformat(timespec="seconds")
        self.registry.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(StrategyDeploymentBlocked, "stale"):
            assert_strategy_deployment_ready(self.settings)

    def test_dry_run_does_not_require_readiness(self):
        from dataclasses import replace

        assert_strategy_deployment_ready(
            replace(self.settings, trading_enabled=False, dry_run=True)
        )

    def test_application_blocks_before_forecast_engine_initialization(self):
        with patch("main.ChronosForecastEngine") as forecast_engine:
            with self.assertRaises(StrategyDeploymentBlocked):
                TradingApplication(self.settings)
        forecast_engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
