"""Deterministic institutional-liquidity H4/M15/M3 breakout analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from execution_agent import Side


@dataclass(frozen=True, slots=True)
class LiquidityBreakoutDecision:
    asset: str
    macro_bias_4h: str
    retail_bait_level: float | None
    whale_target_level_4h: float | None
    entry_price_3m: float | None
    stop_loss_15m: float | None
    take_profit_4h: float | None
    calculated_rrr: float
    side: Side | None
    trade_status: str
    notes: str
    trigger_bar_time: int | None = None

    def payload(
        self,
        *,
        risk_amount_usd: float = 0.0,
        projected_profit_usd: float = 0.0,
        trade_status: str | None = None,
    ) -> dict[str, Any]:
        row = asdict(self)
        row["side"] = self.side.value if self.side else None
        row["calculated_rrr"] = f"1:{self.calculated_rrr:.2f}"
        row["risk_amount_usd"] = round(risk_amount_usd, 2)
        row["projected_profit_usd"] = round(projected_profit_usd, 2)
        if trade_status is not None:
            row["trade_status"] = trade_status
        return row


def _number(row: Any, name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float(getattr(row, name))


def _rows(rates: Iterable[Any]) -> list[Any]:
    return list(rates)


def _average_true_range(rows: list[Any], period: int = 14) -> float:
    if len(rows) < period + 1:
        return 0.0
    values: list[float] = []
    for previous, current in zip(rows[-period - 1 : -1], rows[-period:]):
        high = _number(current, "high")
        low = _number(current, "low")
        previous_close = _number(previous, "close")
        values.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    return sum(values) / len(values)


def _hold(
    asset: str,
    status: str,
    notes: str,
    *,
    bias: str = "NONE",
    bait: float | None = None,
    target: float | None = None,
    entry: float | None = None,
    stop: float | None = None,
    rrr: float = 0.0,
    trigger_bar_time: int | None = None,
) -> LiquidityBreakoutDecision:
    return LiquidityBreakoutDecision(
        asset,
        bias,
        bait,
        target,
        entry,
        stop,
        target,
        rrr,
        None,
        status,
        notes,
        trigger_bar_time,
    )


def daily_lock_reason(
    entries: int,
    net_profit: float,
    target_profit: float,
    maximum_entries: int,
) -> str | None:
    if target_profit > 0 and net_profit >= target_profit:
        return "DAILY_TARGET_REACHED"
    if entries >= maximum_entries:
        return "MAX_DAILY_TRADES_REACHED"
    return None

def effective_daily_entries(reconciled_entries: int, reserved_entries: int) -> int:
    """Avoid double-counting local reservations once MT5 history catches up."""
    return max(reconciled_entries, reserved_entries)



class LiquidityBreakoutEngine:
    """Closed-bar H4 trap, M15 structure, and M3 volume-breakout engine."""

    def __init__(
        self,
        *,
        minimum_rrr: float = 2.5,
        minimum_touches: int = 3,
        volume_expansion: float = 1.2,
        momentum_body_fraction: float = 0.60,
        h4_zone_bars: int = 24,
        h4_history_bars: int = 96,
        m15_structure_bars: int = 20,
        m15_stop_bars: int = 8,
        m3_volume_bars: int = 20,
    ):
        self.minimum_rrr = minimum_rrr
        self.minimum_touches = minimum_touches
        self.volume_expansion = volume_expansion
        self.momentum_body_fraction = momentum_body_fraction
        self.h4_zone_bars = h4_zone_bars
        self.h4_history_bars = h4_history_bars
        self.m15_structure_bars = m15_structure_bars
        self.m15_stop_bars = m15_stop_bars
        self.m3_volume_bars = m3_volume_bars

    def evaluate(
        self,
        asset: str,
        h4_rates: Iterable[Any],
        m15_rates: Iterable[Any],
        m3_rates: Iterable[Any],
        *,
        has_position: bool = False,
    ) -> LiquidityBreakoutDecision:
        h4, m15, m3 = _rows(h4_rates), _rows(m15_rates), _rows(m3_rates)
        required_h4 = self.h4_zone_bars + 20
        if len(h4) < required_h4:
            return _hold(asset, "INSUFFICIENT_H4_BARS", f"Need at least {required_h4} completed H4 bars.")
        if len(m15) < self.m15_structure_bars + 1:
            return _hold(asset, "INSUFFICIENT_M15_BARS", "Not enough completed M15 structure bars.")
        if len(m3) < self.m3_volume_bars + 1:
            return _hold(asset, "INSUFFICIENT_M3_BARS", "Not enough completed M3 volume bars.")

        zone = h4[-self.h4_zone_bars :]
        history = h4[-self.h4_history_bars : -self.h4_zone_bars]
        atr = _average_true_range(h4)
        if atr <= 0:
            return _hold(asset, "INVALID_H4_VOLATILITY", "H4 ATR is non-positive.")

        upper = max(_number(row, "high") for row in zone)
        lower = min(_number(row, "low") for row in zone)
        tolerance = max(atr * 0.20, abs(_number(zone[-1], "close")) * 0.0005)
        upper_touches = sum(_number(row, "high") >= upper - tolerance for row in zone)
        lower_touches = sum(_number(row, "low") <= lower + tolerance for row in zone)
        current = _number(zone[-1], "close")
        midpoint = (upper + lower) / 2.0
        slope = current - _number(zone[0], "close")

        side: Side | None = None
        bias = "NONE"
        bait: float | None = None
        target: float | None = None
        if upper_touches >= self.minimum_touches and current >= midpoint and slope >= 0:
            candidates = sorted(
                {_number(row, "high") for row in history if _number(row, "high") > upper + tolerance}
            )
            if candidates:
                side, bias, bait, target = Side.BUY, "BULLISH_SWEEP", upper, candidates[0]
        elif lower_touches >= self.minimum_touches and current <= midpoint and slope <= 0:
            candidates = sorted(
                {_number(row, "low") for row in history if _number(row, "low") < lower - tolerance},
                reverse=True,
            )
            if candidates:
                side, bias, bait, target = Side.SELL, "BEARISH_SWEEP", lower, candidates[0]

        if side is None:
            touched = max(upper_touches, lower_touches) >= self.minimum_touches
            status = "NO_WHALE_TARGET" if touched else "NO_RETAIL_BAIT_ZONE"
            return _hold(
                asset,
                status,
                "Repeated H4 boundary or fresh external liquidity target was not confirmed.",
            )

        trigger = m3[-1]
        trigger_time = int(_number(trigger, "time"))
        structure = m15[-self.m15_structure_bars - 1 : -1]
        stops = m15[-self.m15_stop_bars :]
        if side is Side.BUY:
            boundary = max(_number(row, "high") for row in structure)
            stop = min(_number(row, "low") for row in stops)
            breakout = _number(trigger, "close") > boundary
            directional = _number(trigger, "close") > _number(trigger, "open")
        else:
            boundary = min(_number(row, "low") for row in structure)
            stop = max(_number(row, "high") for row in stops)
            breakout = _number(trigger, "close") < boundary
            directional = _number(trigger, "close") < _number(trigger, "open")

        entry = _number(trigger, "close")
        if not breakout:
            return _hold(
                asset,
                "M3_NO_BREAKOUT",
                "Latest completed M3 candle has not broken the M15 boundary.",
                bias=bias,
                bait=bait,
                target=target,
                entry=entry,
                stop=stop,
                trigger_bar_time=trigger_time,
            )
        candle_range = _number(trigger, "high") - _number(trigger, "low")
        body = abs(_number(trigger, "close") - _number(trigger, "open"))
        if not directional or candle_range <= 0 or body / candle_range < self.momentum_body_fraction:
            return _hold(
                asset,
                "M3_WEAK_MOMENTUM",
                "Breakout candle lacks one-way directional expansion.",
                bias=bias,
                bait=bait,
                target=target,
                entry=entry,
                stop=stop,
                trigger_bar_time=trigger_time,
            )
        prior_volumes = [_number(row, "tick_volume") for row in m3[-self.m3_volume_bars - 1 : -1]]
        average_volume = sum(prior_volumes) / len(prior_volumes)
        if average_volume <= 0 or _number(trigger, "tick_volume") < average_volume * self.volume_expansion:
            return _hold(
                asset,
                "M3_LOW_VOLUME",
                "Breakout candle volume is below the expansion threshold.",
                bias=bias,
                bait=bait,
                target=target,
                entry=entry,
                stop=stop,
                trigger_bar_time=trigger_time,
            )

        risk = entry - stop if side is Side.BUY else stop - entry
        reward = target - entry if side is Side.BUY else entry - target
        if risk <= 0 or reward <= 0:
            return _hold(
                asset,
                "INVALID_STRUCTURE",
                "M15 stop or H4 target is on the wrong side of entry.",
                bias=bias,
                bait=bait,
                target=target,
                entry=entry,
                stop=stop,
                trigger_bar_time=trigger_time,
            )
        rrr = reward / risk
        if rrr < self.minimum_rrr:
            return _hold(
                asset,
                "INSUFFICIENT_RRR",
                f"Projected reward/risk {rrr:.2f} is below {self.minimum_rrr:.2f}.",
                bias=bias,
                bait=bait,
                target=target,
                entry=entry,
                stop=stop,
                rrr=rrr,
                trigger_bar_time=trigger_time,
            )
        if has_position:
            return _hold(
                asset,
                "POSITION_ALREADY_OPEN",
                "A managed position already exists for this asset.",
                bias=bias,
                bait=bait,
                target=target,
                entry=entry,
                stop=stop,
                rrr=rrr,
                trigger_bar_time=trigger_time,
            )
        return LiquidityBreakoutDecision(
            asset,
            bias,
            bait,
            target,
            entry,
            stop,
            target,
            rrr,
            side,
            "ENTRY_SIGNAL",
            "H4 liquidity target, M15 structure, and M3 momentum/volume confirmed.",
            trigger_time,
        )
