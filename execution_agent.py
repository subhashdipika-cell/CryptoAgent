"""Fail-closed MT5 account, risk, order, and trailing-stop manager."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any
from zoneinfo import ZoneInfo

from config import Settings


LOGGER = logging.getLogger(__name__)


class ExecutionState(Enum):
    DISCONNECTED = auto()
    READY = auto()
    DEGRADED = auto()
    STOPPED = auto()


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    equity: float
    free_margin: float
    margin: float
    leverage: int
    positions: int


@dataclass(frozen=True, slots=True)
class DailyPerformance:
    day: str
    entries: int
    net_profit: float
    target_profit: float

    @property
    def target_reached(self) -> bool:
        return self.net_profit >= self.target_profit


@dataclass(frozen=True, slots=True)
class OrderPlan:
    symbol: str
    side: Side
    volume: float
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    risk_amount: float
    strategy_name: str = "ChronosFinBERT"


@dataclass(frozen=True, slots=True)
class PaperMinimumLotRiskReport:
    """Read-only 1% risk conditions for a fixed 0.01-lot order."""

    symbol: str
    side: Side
    equity: float
    risk_cap_fraction: float
    volume: float
    atr: float
    stop_atr_multiple: float
    stop_distance: float
    broker_minimum_stop_distance: float
    risk_budget: float
    minimum_lot_risk: float
    risk_shortfall: float
    minimum_equity: float
    equity_shortfall: float
    maximum_stop_distance: float
    stop_distance_excess: float
    maximum_atr: float
    atr_excess: float
    fits_risk_cap: bool


class MT5ExecutionAgent:
    def __init__(self, settings: Settings, mt5_module: Any | None = None):
        self.settings = settings
        if mt5_module is None:
            import MetaTrader5 as mt5

            mt5_module = mt5
        self.mt5 = mt5_module
        self.state = ExecutionState.DISCONNECTED

    def connect(self) -> None:
        kwargs: dict[str, Any] = {"timeout": self.settings.mt5_timeout_ms}
        if self.settings.mt5_terminal_path:
            kwargs["path"] = self.settings.mt5_terminal_path
        if self.settings.mt5_login:
            kwargs.update(
                login=self.settings.mt5_login,
                password=self.settings.mt5_password,
                server=self.settings.mt5_server,
            )
        if not self.mt5.initialize(**kwargs):
            self.state = ExecutionState.DEGRADED
            raise ConnectionError(f"MT5 initialize failed: {self.mt5.last_error()}")
        account = self.mt5.account_info()
        if account is None:
            self.shutdown()
            raise ConnectionError(f"MT5 account unavailable: {self.mt5.last_error()}")
        if self.settings.require_demo_account and getattr(account, "trade_mode", None) != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
            self.shutdown()
            raise PermissionError("REQUIRE_DEMO_ACCOUNT is set and the connected account is not DEMO")
        if not self.settings.min_leverage <= account.leverage <= self.settings.max_leverage:
            self.shutdown()
            raise PermissionError(f"account leverage {account.leverage} is outside configured limits")
        self.state = ExecutionState.READY

    def shutdown(self) -> None:
        self.mt5.shutdown()
        self.state = ExecutionState.STOPPED

    def snapshot(self) -> AccountSnapshot:
        account = self.mt5.account_info()
        positions = self.mt5.positions_get()
        if account is None or positions is None:
            self.state = ExecutionState.DEGRADED
            raise RuntimeError(f"unable to read MT5 state: {self.mt5.last_error()}")
        return AccountSnapshot(
            equity=float(account.equity),
            free_margin=float(account.margin_free),
            margin=float(account.margin),
            leverage=int(account.leverage),
            positions=len(positions),
        )

    def daily_performance(self, now: datetime | None = None) -> DailyPerformance:
        """Return current-strategy realized P/L and entries for the configured local day."""
        zone = ZoneInfo(self.settings.liquidity_daily_timezone)
        local_now = now.astimezone(zone) if now is not None else datetime.now(zone)
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        local_end = local_start + timedelta(days=1)
        start_utc, end_utc = local_start.astimezone(timezone.utc), local_end.astimezone(timezone.utc)
        history_start = start_utc - timedelta(days=min(self.settings.history_sync_days, 30))
        deals = self.mt5.history_deals_get(history_start, end_utc)
        if deals is None:
            raise RuntimeError(f"daily MT5 history unavailable: {self.mt5.last_error()}")
        start_ms = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)
        expected_comment = self.settings.order_comment(self.settings.strategy_name)
        strategy_positions = {
            int(getattr(deal, "position_id", 0) or 0)
            for deal in deals
            if int(getattr(deal, "magic", 0) or 0) == self.settings.magic_number
            and int(getattr(deal, "entry", -1)) == 0
            and str(getattr(deal, "comment", "")) == expected_comment
            and int(getattr(deal, "position_id", 0) or 0) > 0
        }
        tracked = [
            deal
            for deal in deals
            if int(getattr(deal, "position_id", 0) or 0) in strategy_positions
            and start_ms <= int(getattr(deal, "time_msc", 0) or 0) < end_ms
        ]
        entries = len({
            int(getattr(deal, "position_id", 0) or 0)
            for deal in tracked
            if int(getattr(deal, "entry", -1)) == 0
        })
        net_profit = sum(
            float(getattr(deal, name, 0.0) or 0.0)
            for deal in tracked
            for name in ("profit", "commission", "swap", "fee")
        )
        account = self.mt5.account_info()
        if account is None:
            raise RuntimeError(f"account unavailable for daily target: {self.mt5.last_error()}")
        active_capital = min(
            self.settings.liquidity_daily_active_capital,
            max(0.0, float(account.equity)),
        )
        return DailyPerformance(
            local_start.date().isoformat(),
            entries,
            net_profit,
            active_capital * self.settings.liquidity_daily_target_fraction,
        )


    def managed_position_symbols(self) -> set[str]:
        positions = self.mt5.positions_get()
        if positions is None:
            raise RuntimeError(f"positions_get failed: {self.mt5.last_error()}")
        return {position.symbol for position in positions if position.magic == self.settings.magic_number}

    def bars(self, symbol: str, timeframe: int, count: int) -> Any:
        if not self.mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select({symbol}) failed: {self.mt5.last_error()}")
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 1, count)  # excludes forming bar
        if rates is None or len(rates) < count:
            received = 0 if rates is None else len(rates)
            raise RuntimeError(f"{symbol} returned {received}/{count} closed bars")
        return rates

    @staticmethod
    def _round_volume(raw: float, minimum: float, maximum: float, step: float) -> float:
        if raw < minimum or step <= 0:
            return 0.0
        units = math.floor((min(raw, maximum) + 1e-12) / step)
        return round(units * step, max(0, int(round(-math.log10(step))) if step < 1 else 0))

    def _loss_per_lot(self, symbol: str, side: Side, entry: float, stop: float, info: Any) -> float:
        order_type = self.mt5.ORDER_TYPE_BUY if side is Side.BUY else self.mt5.ORDER_TYPE_SELL
        calculated = self.mt5.order_calc_profit(order_type, symbol, 1.0, entry, stop)
        if calculated is not None and calculated != 0:
            return abs(float(calculated))
        distance = abs(entry - stop)
        tick_size, tick_value = float(info.trade_tick_size), float(info.trade_tick_value_loss)
        if tick_size <= 0 or tick_value <= 0:
            raise RuntimeError(f"{symbol} has invalid tick value metadata")
        return distance / tick_size * tick_value

    def paper_minimum_lot_risk_report(
        self,
        symbol: str,
        side: Side,
        atr: float,
        *,
        risk_cap_fraction: float = 0.01,
        volume: float = 0.01,
    ) -> PaperMinimumLotRiskReport:
        """Calculate paper-only minimum-equity and stop conditions without sizing an order."""
        account = self.mt5.account_info()
        info = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if account is None or info is None or tick is None:
            raise RuntimeError(f"missing MT5 paper-report metadata for {symbol}: {self.mt5.last_error()}")
        equity = float(account.equity)
        if atr <= 0 or equity <= 0 or not 0 < risk_cap_fraction <= 1 or volume <= 0:
            raise ValueError("ATR, equity, risk cap, and paper volume must be positive")
        volume_min = float(info.volume_min)
        volume_step = float(info.volume_step)
        if volume + 1e-12 < volume_min or volume_step <= 0:
            raise ValueError(f"{symbol} does not support the paper volume {volume:.2f}")
        entry = float(tick.ask if side is Side.BUY else tick.bid)
        sign = 1.0 if side is Side.BUY else -1.0
        digits = int(info.digits)
        stop = round(entry - sign * self.settings.stop_atr_multiple * atr, digits)
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            raise ValueError(f"{symbol} ATR rounds to a zero paper stop distance")
        minimum_lot_risk = volume * self._loss_per_lot(symbol, side, entry, stop, info)
        if minimum_lot_risk <= 0:
            raise ValueError(f"{symbol} has a non-positive 0.01-lot paper risk")
        risk_budget = equity * risk_cap_fraction
        minimum_equity = minimum_lot_risk / risk_cap_fraction
        maximum_stop_distance = stop_distance * risk_budget / minimum_lot_risk
        maximum_atr = maximum_stop_distance / self.settings.stop_atr_multiple
        broker_minimum = float(info.trade_stops_level) * float(info.point)
        return PaperMinimumLotRiskReport(
            symbol=symbol,
            side=side,
            equity=equity,
            risk_cap_fraction=risk_cap_fraction,
            volume=volume,
            atr=float(atr),
            stop_atr_multiple=self.settings.stop_atr_multiple,
            stop_distance=stop_distance,
            broker_minimum_stop_distance=broker_minimum,
            risk_budget=risk_budget,
            minimum_lot_risk=minimum_lot_risk,
            risk_shortfall=max(0.0, minimum_lot_risk - risk_budget),
            minimum_equity=minimum_equity,
            equity_shortfall=max(0.0, minimum_equity - equity),
            maximum_stop_distance=maximum_stop_distance,
            stop_distance_excess=max(0.0, stop_distance - maximum_stop_distance),
            maximum_atr=maximum_atr,
            atr_excess=max(0.0, float(atr) - maximum_atr),
            fits_risk_cap=minimum_lot_risk <= risk_budget + 1e-9,
        )

    def build_structured_order(
        self,
        symbol: str,
        side: Side,
        stop_loss: float,
        take_profit: float,
        atr: float,
        *,
        minimum_rrr: float = 2.5,
    ) -> OrderPlan:
        """Size an order from strategy-defined structure while retaining hard risk caps."""
        account, info, tick = (
            self.mt5.account_info(),
            self.mt5.symbol_info(symbol),
            self.mt5.symbol_info_tick(symbol),
        )
        if account is None or info is None or tick is None:
            raise RuntimeError(f"missing MT5 order metadata for {symbol}: {self.mt5.last_error()}")
        if atr <= 0 or account.equity <= 0:
            raise ValueError("ATR and equity must be positive")
        entry = float(tick.ask if side is Side.BUY else tick.bid)
        digits = int(info.digits)
        stop = round(float(stop_loss), digits)
        target = round(float(take_profit), digits)
        risk = entry - stop if side is Side.BUY else stop - entry
        reward = target - entry if side is Side.BUY else entry - target
        if risk <= 0 or reward <= 0:
            raise ValueError(f"{symbol} structure SL/TP is on the wrong side of market entry")
        rrr = reward / risk
        if rrr < minimum_rrr:
            raise ValueError(
                f"{symbol} broker-price reward/risk {rrr:.2f} is below {minimum_rrr:.2f}"
            )
        min_distance = float(info.trade_stops_level) * float(info.point)
        if risk < min_distance or reward < min_distance:
            raise ValueError(f"{symbol} SL/TP violates broker minimum stop distance")
        risk_budget = float(account.equity) * self.settings.max_risk_fraction
        loss_per_lot = self._loss_per_lot(symbol, side, entry, stop, info)
        volume = self._round_volume(
            risk_budget / loss_per_lot,
            float(info.volume_min),
            float(info.volume_max),
            float(info.volume_step),
        )
        if volume <= 0:
            raise ValueError(f"risk budget is below {symbol}'s minimum lot")
        risk_amount = volume * loss_per_lot
        margin = self.mt5.order_calc_margin(
            self.mt5.ORDER_TYPE_BUY if side is Side.BUY else self.mt5.ORDER_TYPE_SELL,
            symbol,
            volume,
            entry,
        )
        if margin is None or margin > account.margin_free * self.settings.max_margin_fraction:
            raise ValueError(f"{symbol} order exceeds available-margin policy")
        return OrderPlan(
            symbol,
            side,
            volume,
            entry,
            stop,
            target,
            atr,
            risk_amount,
            self.settings.strategy_name,
        )

    def move_positions_to_breakeven(self, minimum_rrr: float = 2.0) -> None:
        """Move this strategy's fixed SL to entry after price reaches the configured R multiple."""
        positions = self.mt5.positions_get()
        if positions is None:
            raise RuntimeError(f"positions_get failed: {self.mt5.last_error()}")
        expected_comment = self.settings.order_comment(self.settings.strategy_name)
        for position in positions:
            if (
                position.magic != self.settings.magic_number
                or str(getattr(position, "comment", "")) != expected_comment
            ):
                continue
            is_buy = position.type == self.mt5.POSITION_TYPE_BUY
            entry, current_sl = float(position.price_open), float(position.sl)
            initial_risk = entry - current_sl if is_buy else current_sl - entry
            if initial_risk <= 0:
                continue
            tick, info = self.mt5.symbol_info_tick(position.symbol), self.mt5.symbol_info(position.symbol)
            if tick is None or info is None:
                continue
            current = float(tick.bid if is_buy else tick.ask)
            profit_distance = current - entry if is_buy else entry - current
            if profit_distance < minimum_rrr * initial_risk:
                continue
            candidate = round(entry, int(info.digits))
            improves = candidate > current_sl if is_buy else candidate < current_sl
            if not improves:
                continue
            request = {
                "action": self.mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": candidate,
                "tp": position.tp,
                "magic": self.settings.magic_number,
            }
            if self.settings.trading_enabled and not self.settings.dry_run:
                result = self.mt5.order_send(request)
                if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
                    LOGGER.error(
                        "Breakeven stop failed for %s: %s",
                        position.ticket,
                        result or self.mt5.last_error(),
                    )
            else:
                LOGGER.info("DRY RUN breakeven update: %s", request)

    def build_order(self, symbol: str, side: Side, atr: float) -> OrderPlan:
        account, info, tick = self.mt5.account_info(), self.mt5.symbol_info(symbol), self.mt5.symbol_info_tick(symbol)
        if account is None or info is None or tick is None:
            raise RuntimeError(f"missing MT5 order metadata for {symbol}: {self.mt5.last_error()}")
        if atr <= 0 or account.equity <= 0:
            raise ValueError("ATR and equity must be positive")
        entry = float(tick.ask if side is Side.BUY else tick.bid)
        sign = 1.0 if side is Side.BUY else -1.0
        stop = entry - sign * self.settings.stop_atr_multiple * atr
        take_profit = entry + sign * self.settings.take_profit_atr_multiple * atr
        digits = int(info.digits)
        stop, take_profit = round(stop, digits), round(take_profit, digits)
        min_distance = float(info.trade_stops_level) * float(info.point)
        if abs(entry - stop) < min_distance or abs(take_profit - entry) < min_distance:
            raise ValueError(f"{symbol} SL/TP violates broker minimum stop distance")
        risk_budget = float(account.equity) * self.settings.max_risk_fraction
        loss_per_lot = self._loss_per_lot(symbol, side, entry, stop, info)
        volume = self._round_volume(
            risk_budget / loss_per_lot,
            float(info.volume_min),
            float(info.volume_max),
            float(info.volume_step),
        )
        if volume <= 0:
            raise ValueError(f"risk budget is below {symbol}'s minimum lot")
        risk_amount = volume * loss_per_lot
        margin = self.mt5.order_calc_margin(
            self.mt5.ORDER_TYPE_BUY if side is Side.BUY else self.mt5.ORDER_TYPE_SELL,
            symbol,
            volume,
            entry,
        )
        if margin is None or margin > account.margin_free * self.settings.max_margin_fraction:
            raise ValueError(f"{symbol} order exceeds available-margin policy")
        return OrderPlan(
            symbol,
            side,
            volume,
            entry,
            stop,
            take_profit,
            atr,
            risk_amount,
            self.settings.strategy_name,
        )

    def submit(self, plan: OrderPlan) -> Any:
        if not self.settings.trading_enabled or self.settings.dry_run:
            LOGGER.info("DRY RUN order plan: %s", plan)
            return None
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": plan.symbol,
            "volume": plan.volume,
            "type": self.mt5.ORDER_TYPE_BUY if plan.side is Side.BUY else self.mt5.ORDER_TYPE_SELL,
            "price": plan.entry,
            "sl": plan.stop_loss,
            "tp": plan.take_profit,
            "deviation": self.settings.max_deviation_points,
            "magic": self.settings.magic_number,
            "comment": self.settings.order_comment(plan.strategy_name),
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        check = self.mt5.order_check(request)
        if check is None or check.retcode not in {0, self.mt5.TRADE_RETCODE_DONE}:
            raise RuntimeError(f"MT5 order check rejected: {check or self.mt5.last_error()}")
        result = self.mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"MT5 order_send failed: {self.mt5.last_error()}")
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            reason = {
                getattr(self.mt5, "TRADE_RETCODE_REQUOTE", -1): "requote",
                getattr(self.mt5, "TRADE_RETCODE_MARKET_CLOSED", -2): "market closed",
                getattr(self.mt5, "TRADE_RETCODE_NO_MONEY", -3): "insufficient margin",
            }.get(result.retcode, result.comment)
            raise RuntimeError(f"MT5 rejected order ({result.retcode}): {reason}")
        return result

    def trail_positions(self, atr_by_symbol: dict[str, float]) -> None:
        positions = self.mt5.positions_get()
        if positions is None:
            raise RuntimeError(f"positions_get failed: {self.mt5.last_error()}")
        for position in positions:
            if position.magic != self.settings.magic_number or position.symbol not in atr_by_symbol:
                continue
            atr = atr_by_symbol[position.symbol]
            tick, info = self.mt5.symbol_info_tick(position.symbol), self.mt5.symbol_info(position.symbol)
            if tick is None or info is None:
                continue
            is_buy = position.type == self.mt5.POSITION_TYPE_BUY
            current = float(tick.bid if is_buy else tick.ask)
            profit_distance = current - position.price_open if is_buy else position.price_open - current
            if profit_distance < self.settings.trailing_trigger_atr * atr:
                continue
            candidate = current - atr * self.settings.trailing_distance_atr if is_buy else current + atr * self.settings.trailing_distance_atr
            candidate = round(candidate, info.digits)
            improves = candidate > position.sl if is_buy else (position.sl == 0 or candidate < position.sl)
            if not improves:
                continue
            request = {
                "action": self.mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": candidate,
                "tp": position.tp,
                "magic": self.settings.magic_number,
            }
            if self.settings.trading_enabled and not self.settings.dry_run:
                result = self.mt5.order_send(request)
                if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
                    LOGGER.error("Trailing stop failed for %s: %s", position.ticket, result or self.mt5.last_error())
            else:
                LOGGER.info("DRY RUN trailing update: %s", request)
