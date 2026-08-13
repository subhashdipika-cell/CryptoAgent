"""Fail-closed MT5 account, risk, order, and trailing-stop manager."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

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
        risk_amount = float(account.equity) * self.settings.max_risk_fraction
        loss_per_lot = self._loss_per_lot(symbol, side, entry, stop, info)
        volume = self._round_volume(
            risk_amount / loss_per_lot,
            float(info.volume_min),
            float(info.volume_max),
            float(info.volume_step),
        )
        if volume <= 0:
            raise ValueError(f"risk budget is below {symbol}'s minimum lot")
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
