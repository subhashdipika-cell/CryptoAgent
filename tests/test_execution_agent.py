import unittest
from types import SimpleNamespace

from config import Settings
from execution_agent import MT5ExecutionAgent, Side


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


class ExecutionTests(unittest.TestCase):
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

    def test_order_plan_has_hard_stops_and_capped_risk(self):
        settings = Settings(dry_run=True, trading_enabled=False)
        plan = MT5ExecutionAgent(settings, FakeMT5()).build_order("BTCUSD", Side.BUY, atr=2.0)
        self.assertLess(plan.stop_loss, plan.entry)
        self.assertGreater(plan.take_profit, plan.entry)
        self.assertLessEqual(plan.risk_amount, 100.0)
        actual_risk = plan.volume * abs(plan.entry - plan.stop_loss) * 100.0
        self.assertLessEqual(actual_risk, plan.risk_amount)


if __name__ == "__main__":
    unittest.main()
