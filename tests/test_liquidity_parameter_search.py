import unittest

from liquidity_parameter_search import (
    parameter_distance,
    research_configuration_hash,
    segment_metrics,
)


class LiquidityParameterSearchTests(unittest.TestCase):
    def test_research_configuration_hash_covers_replay_implementation(self):
        self.assertEqual(len(research_configuration_hash()), 64)

    def test_segment_metrics_include_cost_adjusted_drawdown(self):
        result = segment_metrics([10.0, -5.0, -10.0, 20.0], 1000.0)
        self.assertEqual(result["trades"], 4)
        self.assertEqual(result["net_profit"], 15.0)
        self.assertAlmostEqual(result["profit_factor"], 2.0)
        self.assertAlmostEqual(result["max_drawdown_pct"], 1.5)

    def test_parameter_distance_uses_adjacent_grid_steps(self):
        baseline = {
            "minimum_touches": 2,
            "h4_zone_bars": 12,
            "h4_history_bars": 72,
        }
        neighbor = {**baseline, "h4_zone_bars": 18}
        diagonal = {**neighbor, "h4_history_bars": 96}
        self.assertEqual(parameter_distance(baseline, neighbor), 1)
        self.assertEqual(parameter_distance(baseline, diagonal), 2)


if __name__ == "__main__":
    unittest.main()
