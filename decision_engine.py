"""Governed, asset-specific decision policy for validated forecasts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from execution_agent import Side
from quant_engine import ForecastResult


@dataclass(frozen=True, slots=True)
class AssetDecisionPolicy:
    symbol: str
    model_name: str
    enabled: bool
    confidence_threshold: float
    m15_edge_bps: float
    h1_edge_bps: float
    calibration_trades: int
    holdout_trades: int
    holdout_net_bps: float
    holdout_profit_factor: float


@dataclass(frozen=True, slots=True)
class DecisionResult:
    side: Side | None
    reason: str
    score: float
    required_score: float
    model_name: str

    @property
    def decision(self) -> str:
        return self.side.value if self.side else "HOLD"


class CalibratedDecisionEngine:
    def __init__(self, policy_path: Path):
        self.policy_path = policy_path
        self._policies = self._load(policy_path)

    @staticmethod
    def _load(path: Path) -> dict[str, AssetDecisionPolicy]:
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        policies: dict[str, AssetDecisionPolicy] = {}
        for row in payload.get("policies", []):
            policy = AssetDecisionPolicy(**row)
            policies[policy.symbol] = policy
        return policies

    def evaluate(
        self,
        symbol: str,
        m15: ForecastResult,
        h1: ForecastResult,
        sentiment: float,
        sentiment_degraded: bool,
        has_position: bool = False,
    ) -> DecisionResult:
        policy = self._policies.get(symbol)
        model_name = m15.model_name if m15.model_name == h1.model_name else "MIXED_MODELS"
        if policy is None or not policy.enabled:
            return DecisionResult(None, "UNVALIDATED_MODEL", 0.5, 1.0, model_name)
        if model_name != policy.model_name:
            return DecisionResult(None, "MODEL_POLICY_MISMATCH", 0.5, policy.confidence_threshold, model_name)
        if m15.direction != h1.direction:
            return DecisionResult(None, "TIMEFRAME_DISAGREEMENT", 0.5, policy.confidence_threshold, model_name)

        m15_bull = m15.probability if m15.direction == "BULLISH" else 1.0 - m15.probability
        h1_bull = h1.probability if h1.direction == "BULLISH" else 1.0 - h1.probability
        if sentiment_degraded:
            # Missing sentiment is omitted, not treated as 20% permanent neutrality.
            bullish_score = 0.5 * m15_bull + 0.5 * h1_bull
        else:
            bullish_score = 0.4 * m15_bull + 0.4 * h1_bull + 0.2 * sentiment
        directional_score = bullish_score if m15.direction == "BULLISH" else 1.0 - bullish_score
        sufficient_edge = (
            abs(m15.edge_bps) >= policy.m15_edge_bps
            and abs(h1.edge_bps) >= policy.h1_edge_bps
        )
        if directional_score < policy.confidence_threshold or not sufficient_edge:
            return DecisionResult(
                None, "INSUFFICIENT_EDGE", directional_score,
                policy.confidence_threshold, model_name,
            )
        if has_position:
            return DecisionResult(
                None, "POSITION_ALREADY_OPEN", directional_score,
                policy.confidence_threshold, model_name,
            )
        side = Side.BUY if m15.direction == "BULLISH" else Side.SELL
        return DecisionResult(side, "ENTRY_SIGNAL", directional_score, policy.confidence_threshold, model_name)
