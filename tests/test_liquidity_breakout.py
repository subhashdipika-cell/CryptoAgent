import unittest

from execution_agent import Side
from liquidity_breakout import (
    LiquidityBreakoutEngine,
    daily_lock_reason,
    effective_daily_entries,
)


def bar(
    index,
    *,
    open_price,
    high,
    low,
    close,
    volume=100,
):
    return {
        "time": index,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": volume,
    }


def bullish_setup(*, trigger_volume=150, target=120.0):
    h4 = [
        bar(i, open_price=99.0, high=100.0, low=96.0, close=99.0)
        for i in range(120)
    ]
    h4[80] = bar(80, open_price=110.0, high=target, low=108.0, close=111.0)
    for i in range(96, 120):
        close = 100.0 + 4.0 * (i - 96) / 23
        h4[i] = bar(i, open_price=close - 0.2, high=105.0, low=95.0, close=close)

    m15 = [
        bar(i, open_price=103.0, high=105.0, low=102.0, close=104.0)
        for i in range(30)
    ]
    m3 = [
        bar(i, open_price=104.0, high=104.5, low=103.5, close=104.1, volume=100)
        for i in range(21)
    ]
    m3[-1] = bar(
        20,
        open_price=104.0,
        high=106.2,
        low=103.8,
        close=106.0,
        volume=trigger_volume,
    )
    return h4, m15, m3


class LiquidityBreakoutTests(unittest.TestCase):
    def setUp(self):
        self.engine = LiquidityBreakoutEngine()

    def test_bullish_h4_m15_m3_setup_generates_three_point_five_r_signal(self):
        decision = self.engine.evaluate("XAUUSD+", *bullish_setup())

        self.assertIs(decision.side, Side.BUY)
        self.assertEqual(decision.macro_bias_4h, "BULLISH_SWEEP")
        self.assertEqual(decision.trade_status, "ENTRY_SIGNAL")
        self.assertAlmostEqual(decision.retail_bait_level, 105.0)
        self.assertAlmostEqual(decision.whale_target_level_4h, 120.0)
        self.assertAlmostEqual(decision.entry_price_3m, 106.0)
        self.assertAlmostEqual(decision.stop_loss_15m, 102.0)
        self.assertAlmostEqual(decision.calculated_rrr, 3.5)
        payload = decision.payload(risk_amount_usd=100.0, projected_profit_usd=350.0)
        self.assertEqual(payload["calculated_rrr"], "1:3.50")
        self.assertEqual(payload["risk_amount_usd"], 100.0)

    def test_low_m3_volume_holds(self):
        decision = self.engine.evaluate(
            "XAUUSD+",
            *bullish_setup(trigger_volume=110),
        )

        self.assertIsNone(decision.side)
        self.assertEqual(decision.trade_status, "M3_LOW_VOLUME")

    def test_rrr_below_two_point_five_holds(self):
        decision = self.engine.evaluate(
            "XAUUSD+",
            *bullish_setup(target=114.0),
        )

        self.assertIsNone(decision.side)
        self.assertEqual(decision.trade_status, "INSUFFICIENT_RRR")
        self.assertLess(decision.calculated_rrr, 2.5)

    def test_existing_position_blocks_otherwise_valid_entry(self):
        decision = self.engine.evaluate(
            "XAUUSD+",
            *bullish_setup(),
            has_position=True,
        )

        self.assertIsNone(decision.side)
        self.assertEqual(decision.trade_status, "POSITION_ALREADY_OPEN")

    def test_external_target_buffer_is_isolated_from_zone_touch_tolerance(self):
        h4, m15, m3 = bullish_setup()
        h4[80] = bar(
            80, open_price=105.0, high=106.2, low=104.0, close=105.5
        )
        setup = (h4, m15, m3)
        baseline = self.engine.evaluate("XAUUSD+", *setup)
        relaxed = LiquidityBreakoutEngine(
            external_target_buffer_atr=0.10
        ).evaluate("XAUUSD+", *setup)

        self.assertEqual(baseline.trade_status, "NO_WHALE_TARGET")
        self.assertNotEqual(relaxed.trade_status, "NO_WHALE_TARGET")


    def test_daily_lock_prioritizes_profit_target_and_caps_three_entries(self):
        self.assertEqual(
            daily_lock_reason(2, 250.0, 250.0, 3),
            "DAILY_TARGET_REACHED",
        )
        self.assertEqual(
            daily_lock_reason(3, 0.0, 250.0, 3),
            "MAX_DAILY_TRADES_REACHED",
        )
        self.assertIsNone(daily_lock_reason(2, 249.99, 250.0, 3))
        self.assertEqual(effective_daily_entries(1, 2), 2)
        self.assertEqual(effective_daily_entries(2, 2), 2)

if __name__ == "__main__":
    unittest.main()
