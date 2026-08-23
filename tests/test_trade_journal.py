import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from config import Settings
from performance_report import (
    FORWARD_EVIDENCE_AVAILABLE,
    INSUFFICIENT_FORWARD_EVIDENCE,
    completed_trades,
    forward_evidence,
    generate_report,
    metrics,
)
from execution_agent import PaperMinimumLotRiskReport, Side
from trade_journal import TradeJournal


class FakeHistoryMT5:
    def __init__(self):
        self.orders = [
            SimpleNamespace(
                ticket=100,
                time_msc=1_700_000_000_000,
                position_id=100,
                symbol="BTCUSD",
                type=0,
                state=4,
                volume_initial=0.01,
                volume_current=0.0,
                price_open=100.0,
                sl=98.0,
                tp=104.0,
                price_current=100.0,
                magic=26081301,
                comment="CryptoAgent|ChronosFinBERT",
                reason=3,
                external_id="",
            ),
            SimpleNamespace(
                ticket=999,
                time_msc=1_700_000_000_000,
                position_id=999,
                symbol="EURUSD",
                magic=7,
                comment="unrelated",
            ),
        ]
        self.deals = [
            SimpleNamespace(
                ticket=200,
                order=100,
                position_id=100,
                time_msc=1_700_000_000_000,
                symbol="BTCUSD",
                type=0,
                entry=0,
                volume=0.01,
                price=100.0,
                profit=0.0,
                commission=-0.10,
                swap=0.0,
                fee=0.0,
                magic=26081301,
                comment="CryptoAgent|ChronosFinBERT",
                reason=3,
                external_id="",
            ),
            SimpleNamespace(
                ticket=201,
                order=101,
                position_id=100,
                time_msc=1_700_000_600_000,
                symbol="BTCUSD",
                type=1,
                entry=1,
                volume=0.01,
                price=102.0,
                profit=2.0,
                commission=-0.10,
                swap=-0.05,
                fee=0.0,
                magic=26081301,
                comment="[tp]",
                reason=5,
                external_id="",
            ),
            SimpleNamespace(
                ticket=999,
                order=999,
                position_id=999,
                time_msc=1_700_000_000_000,
                symbol="EURUSD",
                magic=7,
                comment="unrelated",
            ),
        ]

    def history_orders_get(self, _start, _end):
        return self.orders

    def history_deals_get(self, _start, _end):
        return self.deals

    def last_error(self):
        return (0, "ok")


class TradeJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = Settings(journal_path=root / "journal.db", report_dir=root / "reports")
        self.account = SimpleNamespace(login=1234, server="Broker-Demo")

    def tearDown(self):
        self.temp.cleanup()

    def test_sync_is_filtered_and_idempotent(self):
        journal = TradeJournal(self.settings)
        terminal = FakeHistoryMT5()
        self.assertEqual(journal.sync_mt5_history(terminal, self.account), {"orders": 1, "deals": 2})
        journal.sync_mt5_history(terminal, self.account)
        self.assertEqual(len(journal.rows("mt5_orders")), 1)
        self.assertEqual(len(journal.rows("mt5_deals")), 2)

    def test_completed_trade_includes_costs_and_exit_reason(self):
        journal = TradeJournal(self.settings)
        journal.sync_mt5_history(FakeHistoryMT5(), self.account)
        trades = completed_trades(journal.rows("mt5_deals"), self.settings)
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0].net_profit, 1.75)
        self.assertEqual(trades[0].strategy, "ChronosFinBERT")
        self.assertEqual(trades[0].exit_reason, "TAKE_PROFIT")
        self.assertAlmostEqual(metrics(trades)["profit_factor"], float("inf"))

    def test_generates_csv_and_html_artifacts(self):
        journal = TradeJournal(self.settings)
        journal.sync_mt5_history(FakeHistoryMT5(), self.account)
        exports = generate_report(self.settings)
        self.assertTrue(exports["html"].is_file())
        self.assertTrue(exports["completed_trades"].is_file())
        self.assertIn("ChronosFinBERT", exports["html"].read_text(encoding="utf-8"))

    def test_order_plan_rejection_persists_one_percent_shortfall(self):
        journal = TradeJournal(self.settings)
        report = PaperMinimumLotRiskReport(
            "XAUUSD+", Side.BUY, 10_000.0, 0.01, 0.01, 100.0, 1.5,
            150.0, 0.1, 100.0, 150.0, 50.0, 15_000.0, 5_000.0,
            100.0, 50.0, 100.0 / 1.5, 100.0 / 3.0, False,
        )
        journal.record_order_plan_rejection(
            self.account,
            "XAUUSD+",
            Side.BUY,
            ValueError("risk budget is below XAUUSD+'s minimum lot"),
            report,
        )
        rows = journal.rows("order_plan_rejections")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "XAUUSD+")
        self.assertAlmostEqual(rows[0]["risk_shortfall"], 50.0)
        self.assertAlmostEqual(rows[0]["equity_shortfall"], 5_000.0)
        self.assertAlmostEqual(rows[0]["maximum_atr"], 100.0 / 1.5)



    def _record_account_mode(self, journal, trade_mode):
        journal.record_account(
            SimpleNamespace(
                login=1234,
                server="Broker-Demo",
                trade_mode=trade_mode,
                balance=1000.0,
            ),
            SimpleNamespace(
                equity=1000.0,
                margin=0.0,
                free_margin=1000.0,
                leverage=100,
                positions=0,
            ),
        )

    def test_forward_evidence_uses_demo_post_activation_trades_and_costs(self):
        journal = TradeJournal(self.settings)
        self._record_account_mode(journal, trade_mode=0)
        journal.sync_mt5_history(FakeHistoryMT5(), self.account)
        trades = completed_trades(journal.rows("mt5_deals"), self.settings)
        policy = {
            "policies": [
                {"symbol": "BTCUSD", "activated_at": "2023-11-14T22:00:00Z"}
            ]
        }

        row = forward_evidence(
            trades,
            journal.rows("account_snapshots"),
            policy,
            minimum_trades=1,
            expert_ids={26081301},
        )[0]

        self.assertEqual(row["evidence_state"], FORWARD_EVIDENCE_AVAILABLE)
        self.assertEqual(row["sample_size"], 1)
        self.assertAlmostEqual(row["net_profit_after_costs"], 1.75)
        self.assertAlmostEqual(row["win_rate_pct"], 100.0)
        self.assertAlmostEqual(row["profit_factor"], float("inf"))
        self.assertAlmostEqual(row["max_drawdown"], 0.0)

        wrong_expert = forward_evidence(
            trades,
            journal.rows("account_snapshots"),
            policy,
            minimum_trades=1,
            expert_ids={7},
        )[0]

        self.assertEqual(wrong_expert["sample_size"], 0)

    def test_forward_evidence_is_insufficient_before_activation_or_without_demo_proof(self):
        journal = TradeJournal(self.settings)
        self._record_account_mode(journal, trade_mode=1)
        journal.sync_mt5_history(FakeHistoryMT5(), self.account)
        trades = completed_trades(journal.rows("mt5_deals"), self.settings)
        policy = {
            "policies": [
                {"symbol": "BTCUSD", "activated_at": "2023-11-14T23:00:00Z"}
            ]
        }

        row = forward_evidence(trades, journal.rows("account_snapshots"), policy)[0]

        self.assertEqual(row["evidence_state"], INSUFFICIENT_FORWARD_EVIDENCE)
        self.assertEqual(row["sample_size"], 0)
        self.assertEqual(row["minimum_sample_size"], 30)

    def test_forward_evidence_honors_policy_rejection_window(self):
        journal = TradeJournal(self.settings)
        self._record_account_mode(journal, trade_mode=0)
        journal.sync_mt5_history(FakeHistoryMT5(), self.account)
        trades = completed_trades(journal.rows("mt5_deals"), self.settings)
        policy = {
            "policies": [{"symbol": "BTCUSD"}],
            "approval_audit": [
                {
                    "symbol": "BTCUSD",
                    "action": "MANUAL_APPROVAL",
                    "approved_at": "2023-11-14T21:00:00Z",
                },
                {
                    "symbol": "BTCUSD",
                    "action": "BACKTEST_REJECTION",
                    "rejected_at": "2023-11-14T22:00:00Z",
                },
            ],
        }

        row = forward_evidence(
            trades,
            journal.rows("account_snapshots"),
            policy,
            minimum_trades=1,
        )[0]

        self.assertEqual(row["evidence_state"], INSUFFICIENT_FORWARD_EVIDENCE)
        self.assertEqual(row["sample_size"], 0)
        self.assertEqual(row["deactivation_at"], "2023-11-14T22:00:00+00:00")

if __name__ == "__main__":
    unittest.main()
