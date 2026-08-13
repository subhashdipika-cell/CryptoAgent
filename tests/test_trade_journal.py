import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from config import Settings
from performance_report import completed_trades, generate_report, metrics
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


if __name__ == "__main__":
    unittest.main()
