import csv
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import Settings
from strategy_readiness import (
    DEMO_READY,
    StrategyDeploymentBlocked,
    assert_strategy_deployment_ready,
    generate_registry,
    strategy_configuration_hash,
    write_dashboard,
)


class StrategyReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.policy = root / "asset_policy.json"
        self.registry = root / "strategy_readiness.json"
        self.reports = root / "reports"
        self.reports.mkdir()
        self.policy.write_text(
            json.dumps(
                {
                    "policies": [
                        {
                            "symbol": "BTCUSD",
                            "enabled": True,
                            "approved": True,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.settings = Settings(
            symbols=("BTCUSD",),
            decision_policy_path=self.policy,
            strategy_readiness_path=self.registry,
            report_dir=self.reports,
            predictive_mode="calibrated",
            strategy_name="AssetCalibrated",
            trading_enabled=True,
            dry_run=False,
            mt5_terminal_path="D:/test/terminal64.exe",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _write_proof(self, age_minutes: int = 0) -> None:
        (self.reports / "reconciliation_status.json").write_text(
            json.dumps(
                {
                    "status": "SUCCESS",
                    "synced_at": (
                        datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
                    ).isoformat(timespec="seconds"),
                    "server": "Broker-Demo",
                    "trade_mode": 0,
                    "require_demo_account": True,
                    "orders": 1,
                    "deals": 1,
                }
            ),
            encoding="utf-8",
        )

    def _write_evidence(self, **overrides) -> None:
        row = {
            "symbol": "BTCUSD",
            "sample_size": 30,
            "sessions": 10,
            "net_profit_after_costs": 100.0,
            "expectancy_after_costs": 3.33,
            "profit_factor": 1.30,
            "max_drawdown_pct": 4.0,
        }
        row.update(overrides)
        with (self.reports / "forward_evidence.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_every_gate_passes_before_demo_routing(self):
        self._write_proof()
        self._write_evidence()
        registry = generate_registry(self.settings)
        self.assertEqual(registry["strategies"][0]["status"], DEMO_READY)
        write_dashboard(registry, self.settings)
        assert_strategy_deployment_ready(self.settings)

    def test_weak_profit_factor_blocks_routing(self):
        self._write_proof()
        self._write_evidence(profit_factor=1.19)
        registry = generate_registry(self.settings)
        write_dashboard(registry, self.settings)
        with self.assertRaisesRegex(StrategyDeploymentBlocked, "PROFIT_FACTOR"):
            assert_strategy_deployment_ready(self.settings)

    def test_stale_reconciliation_blocks_routing(self):
        self._write_proof(age_minutes=91)
        self._write_evidence()
        registry = generate_registry(self.settings)
        self.assertFalse(registry["reconciliation"]["fresh_demo_reconciliation"])

    def test_configuration_change_invalidates_ready_registry(self):
        self._write_proof()
        self._write_evidence()
        write_dashboard(generate_registry(self.settings), self.settings)
        changed = replace(self.settings, max_risk_fraction=0.01)
        self.assertNotEqual(
            strategy_configuration_hash(self.settings),
            strategy_configuration_hash(changed),
        )
        with self.assertRaisesRegex(StrategyDeploymentBlocked, "configuration changed"):
            assert_strategy_deployment_ready(changed)

    def test_live_capable_configuration_is_never_authorized(self):
        unsafe = replace(self.settings, require_demo_account=False)
        with self.assertRaisesRegex(StrategyDeploymentBlocked, "DEMO"):
            assert_strategy_deployment_ready(unsafe)

    def test_dry_run_does_not_require_deployment_readiness(self):
        safe = replace(self.settings, trading_enabled=False, dry_run=True)
        assert_strategy_deployment_ready(safe)


if __name__ == "__main__":
    unittest.main()
