import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from config import Settings
from execution_agent import MT5ExecutionAgent, OrderPlan, Side


class FakeMT5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    def account_info(self):
        return SimpleNamespace(equity=10_000.0, margin_free=8_000.0)

    def symbol_info(self, _symbol):
        return SimpleNamespace(
            digits=2,
            point=0.01,
            trade_stops_level=10,
            trade_tick_size=0.01,
            trade_tick_value_loss=1.0,
            volume_min=0.01,
            volume_max=10.0,
            volume_step=0.01,
        )

    def symbol_info_tick(self, _symbol):
        return SimpleNamespace(ask=100.0, bid=99.98)

    def order_calc_profit(self, _kind, _symbol, _volume, entry, stop):
        return abs(entry - stop) * 100.0

    def order_calc_margin(self, *_args):
        return 100.0

    def last_error(self):
        return (0, "ok")


class FakeSubmitMT5(FakeMT5):
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self.request = None

    def order_check(self, request):
        self.request = request
        return SimpleNamespace(retcode=0)

    def order_send(self, request):
        self.request = request
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, comment="Request executed")

class FakeLiquidityMT5(FakeSubmitMT5):
    TRADE_ACTION_SLTP = 2
    POSITION_TYPE_BUY = 0

    def __init__(self):
        super().__init__()
        self.deals = []
        self.positions = []

    def history_deals_get(self, _start, _end):
        return self.deals

    def positions_get(self):
        return self.positions

    def symbol_info_tick(self, _symbol):
        return SimpleNamespace(ask=104.02, bid=104.0)



class ExecutionTests(unittest.TestCase):
    def test_submitted_order_identifies_cryptoagent(self):
        settings = Settings(
            trading_enabled=True,
            dry_run=False,
            mt5_login=12345,
            magic_number=26081301,
            application_name="CryptoAgent",
            strategy_name="ChronosFinBERT",
        )
        fake = FakeSubmitMT5()
        agent = MT5ExecutionAgent(settings, fake)
        plan = OrderPlan(
            "BTCUSD",
            Side.BUY,
            0.01,
            100.0,
            98.0,
            104.0,
            1.0,
            2.0,
            "ChronosFinBERT",
        )
        agent.submit(plan)
        self.assertEqual(fake.request["magic"], 26081301)
        self.assertEqual(fake.request["comment"], "CryptoAgent|ChronosFinBERT")

    def test_order_comment_rejects_mt5_overflow(self):
        settings = Settings(application_name="CryptoAgent", strategy_name="x" * 30)
        with self.assertRaisesRegex(ValueError, "exceeds 31"):
            settings.validate()

    def test_routing_can_attach_to_authenticated_demo_terminal(self):
        settings = Settings(
            trading_enabled=True,
            dry_run=False,
            require_demo_account=True,
            mt5_terminal_path=r"D:\MT5IntelliTrade\terminal64.exe",
        )
        settings.validate()

    def test_broker_symbols_can_be_overridden(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"MT5_BTC_SYMBOL": "BTCUSD", "MT5_XAU_SYMBOL": "XAUUSD+"}):
            self.assertEqual(Settings().symbols, ("BTCUSD", "XAUUSD+"))

    def test_default_gold_symbol_matches_demo_broker(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings().symbols, ("BTCUSD", "XAUUSD+"))

    def test_order_plan_has_hard_stops_and_capped_risk(self):
        settings = Settings(dry_run=True, trading_enabled=False)
        plan = MT5ExecutionAgent(settings, FakeMT5()).build_order("BTCUSD", Side.BUY, atr=2.0)
        self.assertLess(plan.stop_loss, plan.entry)
        self.assertGreater(plan.take_profit, plan.entry)
        self.assertLessEqual(plan.risk_amount, 200.0)
        actual_risk = plan.volume * abs(plan.entry - plan.stop_loss) * 100.0
        self.assertLessEqual(actual_risk, plan.risk_amount)

    def test_xau_paper_report_calculates_one_percent_shortfalls_and_stop_limits(self):
        settings = Settings(dry_run=True, trading_enabled=False, max_risk_fraction=0.02)
        agent = MT5ExecutionAgent(settings, FakeMT5())
        report = agent.paper_minimum_lot_risk_report("XAUUSD+", Side.BUY, atr=100.0)
        self.assertEqual(report.risk_cap_fraction, 0.01)
        self.assertEqual(report.volume, 0.01)
        self.assertAlmostEqual(report.risk_budget, 100.0)
        self.assertAlmostEqual(report.minimum_lot_risk, 150.0)
        self.assertAlmostEqual(report.risk_shortfall, 50.0)
        self.assertAlmostEqual(report.minimum_equity, 15_000.0)
        self.assertAlmostEqual(report.equity_shortfall, 5_000.0)
        self.assertAlmostEqual(report.maximum_stop_distance, 100.0)
        self.assertAlmostEqual(report.maximum_atr, 100.0 / 1.5)
        self.assertFalse(report.fits_risk_cap)
    def test_structured_order_revalidates_rrr_and_preserves_two_percent_cap(self):
        settings = Settings(
            dry_run=True,
            trading_enabled=False,
            max_risk_fraction=0.02,
            strategy_name="LiquidityBreakout",
        )
        agent = MT5ExecutionAgent(settings, FakeMT5())
        plan = agent.build_structured_order(
            "XAUUSD+",
            Side.BUY,
            stop_loss=98.0,
            take_profit=105.0,
            atr=1.0,
            minimum_rrr=2.5,
        )
        self.assertAlmostEqual(plan.entry, 100.0)
        self.assertAlmostEqual(plan.stop_loss, 98.0)
        self.assertAlmostEqual(plan.take_profit, 105.0)
        self.assertLessEqual(plan.risk_amount, 200.0)
        with self.assertRaisesRegex(ValueError, "reward/risk"):
            agent.build_structured_order(
                "XAUUSD+",
                Side.BUY,
                stop_loss=98.0,
                take_profit=104.9,
                atr=1.0,
                minimum_rrr=2.5,
            )

    def test_daily_performance_counts_current_magic_and_costs(self):
        settings = Settings(
            strategy_name="LiquidityBreakout",
            liquidity_daily_timezone="UTC",
            liquidity_daily_active_capital=1000.0,
            liquidity_daily_target_fraction=0.25,
        )
        fake = FakeLiquidityMT5()
        now = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)
        timestamp = int(now.timestamp() * 1000)
        fake.deals = [
            SimpleNamespace(
                magic=settings.magic_number,
                position_id=1,
                comment="CryptoAgent|LiquidityBreakout",
                time_msc=timestamp,
                entry=0,
                profit=0.0,
                commission=-1.0,
                swap=0.0,
                fee=0.0,
            ),
            SimpleNamespace(
                magic=settings.magic_number,
                position_id=1,
                comment="CryptoAgent|LiquidityBreakout",
                time_msc=timestamp,
                entry=0,
                profit=0.0,
                commission=0.0,
                swap=0.0,
                fee=0.0,
            ),
            SimpleNamespace(
                magic=settings.magic_number,
                position_id=1,
                comment="[tp]",
                time_msc=timestamp,
                entry=1,
                profit=252.0,
                commission=-1.0,
                swap=0.0,
                fee=0.0,
            ),
            SimpleNamespace(
                magic=settings.magic_number,
                position_id=2,
                comment="unrelated",
                time_msc=timestamp,
                entry=0,
                profit=1000.0,
                commission=0.0,
                swap=0.0,
                fee=0.0,
            ),
        ]

        daily = MT5ExecutionAgent(settings, fake).daily_performance(now)

        self.assertEqual(daily.entries, 1)
        self.assertAlmostEqual(daily.net_profit, 250.0)
        self.assertAlmostEqual(daily.target_profit, 250.0)
        self.assertTrue(daily.target_reached)

    def test_liquidity_position_moves_to_breakeven_only_at_two_r(self):
        settings = Settings(
            trading_enabled=True,
            dry_run=False,
            mt5_terminal_path=r"D:\MT5IntelliTrade\terminal64.exe",
            strategy_name="LiquidityBreakout",
        )
        fake = FakeLiquidityMT5()
        fake.positions = [
            SimpleNamespace(
                magic=settings.magic_number,
                symbol="XAUUSD+",
                comment="CryptoAgent|LiquidityBreakout",
                type=fake.POSITION_TYPE_BUY,
                price_open=100.0,
                sl=98.0,
                tp=106.0,
                ticket=123,
            )
        ]

        MT5ExecutionAgent(settings, fake).move_positions_to_breakeven()

        self.assertEqual(fake.request["action"], fake.TRADE_ACTION_SLTP)
        self.assertEqual(fake.request["position"], 123)
        self.assertAlmostEqual(fake.request["sl"], 100.0)
        self.assertAlmostEqual(fake.request["tp"], 106.0)


    def test_paper_cap_does_not_change_existing_runtime_sizing_limit(self):
        settings = Settings(dry_run=True, trading_enabled=False, max_risk_fraction=0.02)
        agent = MT5ExecutionAgent(settings, FakeMT5())
        report = agent.paper_minimum_lot_risk_report("XAUUSD+", Side.BUY, atr=100.0)
        plan = agent.build_order("XAUUSD+", Side.BUY, atr=100.0)
        self.assertFalse(report.fits_risk_cap)
        self.assertEqual(settings.max_risk_fraction, 0.02)
        self.assertEqual(plan.volume, 0.01)
        self.assertLessEqual(plan.risk_amount, 200.0)


    def test_liquidity_configuration_fails_closed_below_rrr_or_above_trade_cap(self):
        with self.assertRaisesRegex(ValueError, "LIQUIDITY_MIN_RRR"):
            Settings(
                strategy_mode="liquidity_breakout",
                liquidity_min_rrr=2.49,
            ).validate()
        with self.assertRaisesRegex(ValueError, "LIQUIDITY_MAX_TRADES_PER_DAY"):
            Settings(
                strategy_mode="liquidity_breakout",
                liquidity_max_trades_per_day=4,
            ).validate()

if __name__ == "__main__":
    unittest.main()
