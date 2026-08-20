"""Asynchronous coordinator for local forecasts, cloud sentiment, and MT5."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from logging.handlers import RotatingFileHandler

from config import LOG_DIR, SETTINGS, Settings
from asset_predictive_engine import DedicatedAssetForecastEngine
from decision_engine import CalibratedDecisionEngine, DecisionResult
from execution_agent import MT5ExecutionAgent, Side
from quant_engine import ChronosForecastEngine, ForecastResult, close_prices, true_range_atr
from revalidation_scheduler import RevalidationScheduler
from sentiment_engine import SentimentEngine, SentimentResult
from trade_journal import TradeJournal


LOGGER = logging.getLogger("crypto_agent")


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = RotatingFileHandler(LOG_DIR / "trading.log", maxBytes=5_000_000, backupCount=5)
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler], force=True)


def combined_side(m15: ForecastResult, h1: ForecastResult, sentiment: float, threshold: float) -> Side | None:
    if m15.direction != h1.direction:
        return None
    m15_bull = m15.probability if m15.direction == "BULLISH" else 1.0 - m15.probability
    h1_bull = h1.probability if h1.direction == "BULLISH" else 1.0 - h1.probability
    bullish = 0.4 * m15_bull + 0.4 * h1_bull + 0.2 * sentiment
    if bullish >= threshold:
        return Side.BUY
    if bullish <= 1.0 - threshold:
        return Side.SELL
    return None


class TradingApplication:
    def __init__(self, settings: Settings = SETTINGS):
        settings.validate()
        self.settings = settings
        self.quant = ChronosForecastEngine(settings)
        self.dedicated_quant = DedicatedAssetForecastEngine(settings)
        self.decisions = CalibratedDecisionEngine(settings.decision_policy_path)
        self.execution = MT5ExecutionAgent(settings)
        self.journal = TradeJournal(settings)
        self.revalidation = RevalidationScheduler(settings)
        self.stop_event = asyncio.Event()
        self._sentiment = SentimentResult(0.5, 0, degraded=True)
        self._sentiment_at = 0.0

    def request_stop(self) -> None:
        self.stop_event.set()

    async def _refresh_sentiment(self, engine: SentimentEngine) -> None:
        if time.monotonic() - self._sentiment_at < self.settings.sentiment_refresh_seconds:
            return
        self._sentiment = await engine.score()
        self._sentiment_at = time.monotonic()
        LOGGER.info(
            "sentiment score=%.3f headlines=%d degraded=%s",
            self._sentiment.score,
            self._sentiment.headline_count,
            self._sentiment.degraded,
        )

    async def _process_symbol(
        self,
        symbol: str,
        managed: set[str],
        account: object,
        snapshot: object,
    ) -> tuple[str, float, tuple[int, ...]]:
        def load_rates() -> tuple[object, object]:
            # MetaTrader5's Python bridge is treated as single-threaded.
            return (
                self.execution.bars(symbol, self.execution.mt5.TIMEFRAME_M15, self.settings.bar_count),
                self.execution.bars(symbol, self.execution.mt5.TIMEFRAME_H1, self.settings.bar_count),
            )

        m15_rates, h1_rates = await asyncio.to_thread(load_rates)
        atr = true_range_atr(m15_rates, self.settings.atr_period)
        # A single model instance is intentionally called serially; torch pipelines are not assumed thread-safe.
        try:
            direct_m15 = await asyncio.to_thread(
                self.dedicated_quant.forecast, symbol, m15_rates, "15min"
            )
            direct_h1 = await asyncio.to_thread(
                self.dedicated_quant.forecast, symbol, h1_rates, "1h"
            )
        except Exception:
            if self.settings.predictive_mode in {"calibrated", "dedicated"}:
                raise
            LOGGER.exception("%s dedicated shadow forecast unavailable", symbol)
            direct_m15 = direct_h1 = None
        if self.settings.predictive_mode in {"calibrated", "dedicated"}:
            m15, h1 = direct_m15, direct_h1
        else:
            # Shadow mode collects comparable live forecasts without changing orders.
            m15 = await asyncio.to_thread(self.quant.forecast, close_prices(m15_rates), "15min")
            h1 = await asyncio.to_thread(self.quant.forecast, close_prices(h1_rates), "1h")
            if direct_m15 is not None and direct_h1 is not None:
                await asyncio.to_thread(
                    self.journal.record_model_forecast,
                    account, symbol, "M15", direct_m15, "shadow",
                )
                await asyncio.to_thread(
                    self.journal.record_model_forecast,
                    account, symbol, "H1", direct_h1, "shadow",
                )
                LOGGER.info(
                    "%s shadow %s M15=%s/%.3f/%+.1fbp H1=%s/%.3f/%+.1fbp",
                    symbol,
                    direct_m15.model_name,
                    direct_m15.direction,
                    direct_m15.probability,
                    direct_m15.edge_bps,
                    direct_h1.direction,
                    direct_h1.probability,
                    direct_h1.edge_bps,
                )
        if self.settings.predictive_mode in {"calibrated", "dedicated"}:
            await asyncio.to_thread(
                self.journal.record_model_forecast,
                account, symbol, "M15", m15, "active",
            )
            await asyncio.to_thread(
                self.journal.record_model_forecast,
                account, symbol, "H1", h1, "active",
            )
        if self.settings.predictive_mode in {"calibrated", "dedicated"}:
            outcome = self.decisions.evaluate(
                symbol, m15, h1, self._sentiment.score, self._sentiment.degraded,
                has_position=symbol in managed,
            )
        else:
            side = combined_side(m15, h1, self._sentiment.score, self.settings.signal_threshold)
            outcome = DecisionResult(
                side,
                "ENTRY_SIGNAL" if side else (
                    "TIMEFRAME_DISAGREEMENT" if m15.direction != h1.direction else "INSUFFICIENT_EDGE"
                ),
                0.5,
                self.settings.signal_threshold,
                m15.model_name,
            )
        plan = None
        rejection_report = None
        rejection_report_error = None
        if outcome.side:
            attempted_side = outcome.side
            try:
                plan = await asyncio.to_thread(
                    self.execution.build_order, symbol, outcome.side, atr
                )
            except Exception as error:
                try:
                    rejection_report = await asyncio.to_thread(
                        self.execution.paper_minimum_lot_risk_report,
                        symbol,
                        attempted_side,
                        atr,
                    )
                except Exception as report_error:
                    rejection_report_error = report_error
                    LOGGER.warning(
                        "%s entry plan rejected: %s; 1%% paper shortfall unavailable: %s",
                        symbol, error, report_error,
                    )
                else:
                    LOGGER.warning(
                        "%s entry plan rejected: %s; 1%% paper risk shortfall=%.2f "
                        "equity shortfall=%.2f minimum equity=%.2f max stop=%.5f max ATR=%.5f",
                        symbol,
                        error,
                        rejection_report.risk_shortfall,
                        rejection_report.equity_shortfall,
                        rejection_report.minimum_equity,
                        rejection_report.maximum_stop_distance,
                        rejection_report.maximum_atr,
                    )
                await asyncio.to_thread(
                    self.journal.record_order_plan_rejection,
                    account,
                    symbol,
                    attempted_side,
                    error,
                    rejection_report,
                    rejection_report_error,
                )
                outcome = DecisionResult(
                    None, "ORDER_PLAN_REJECTED", outcome.score,
                    outcome.required_score, outcome.model_name,
                )
        await asyncio.to_thread(
            self.journal.record_signal,
            account=account,
            snapshot=snapshot,
            symbol=symbol,
            strategy=self.settings.strategy_name,
            m15=m15,
            h1=h1,
            sentiment=self._sentiment,
            atr=atr,
            decision=outcome.decision,
            decision_reason=outcome.reason,
            calibrated_score=outcome.score,
            required_score=outcome.required_score,
            active_model=outcome.model_name,
        )
        LOGGER.info(
            "%s model=%s M15=%s/%.3f H1=%s/%.3f ATR=%.5f decision=%s reason=%s score=%.3f required=%.3f",
            symbol,
            m15.model_name,
            m15.direction,
            m15.probability,
            h1.direction,
            h1.probability,
            atr,
            outcome.decision,
            outcome.reason,
            outcome.score,
            outcome.required_score,
        )
        if plan is not None:
            try:
                result = await asyncio.to_thread(self.execution.submit, plan)
            except Exception as error:
                await asyncio.to_thread(self.journal.record_submission, account, plan, None, error)
                raise
            else:
                await asyncio.to_thread(self.journal.record_submission, account, plan, result)
        return symbol, atr, tuple(int(row["time"]) for row in m15_rates)

    async def _run_cycle(
        self, sentiment: SentimentEngine, allow_revalidation: bool = True
    ) -> None:
        await self._refresh_sentiment(sentiment)
        snapshot = await asyncio.to_thread(self.execution.snapshot)
        account = self.execution.mt5.account_info()
        if account is None:
            raise RuntimeError(f"unable to read MT5 account for journal: {self.execution.mt5.last_error()}")
        await asyncio.to_thread(self.journal.record_account, account, snapshot)
        managed = await asyncio.to_thread(self.execution.managed_position_symbols)
        LOGGER.info(
            "account equity=%.2f free_margin=%.2f positions=%d",
            snapshot.equity,
            snapshot.free_margin,
            snapshot.positions,
        )
        atr_by_symbol: dict[str, float] = {}
        completed_m15: dict[str, tuple[int, ...]] = {}
        for symbol in self.settings.symbols:
            try:
                name, atr, m15_times = await self._process_symbol(
                    symbol, managed, account, snapshot
                )
                atr_by_symbol[name] = atr
                completed_m15[name] = m15_times
            except Exception:
                LOGGER.exception("symbol cycle failed for %s", symbol)
        await asyncio.to_thread(self.execution.trail_positions, atr_by_symbol)
        if allow_revalidation:
            self.revalidation.observe(completed_m15)
        synced = await asyncio.to_thread(
            self.journal.sync_mt5_history,
            self.execution.mt5,
            account,
        )
        LOGGER.info("journal reconciled orders=%d deals=%d", synced["orders"], synced["deals"])

    async def run_once(self) -> None:
        """Run one complete dry-run cycle for deployment verification."""
        if self.settings.trading_enabled or not self.settings.dry_run:
            raise PermissionError("run_once is restricted to dry-run mode")
        if self.settings.predictive_mode == "shadow":
            await asyncio.to_thread(self.quant.load)
        await asyncio.to_thread(self.execution.connect)
        account = self.execution.mt5.account_info()
        if account is None:
            raise RuntimeError("MT5 account unavailable after connection")
        await asyncio.to_thread(self.journal.sync_mt5_history, self.execution.mt5, account)
        try:
            async with SentimentEngine(self.settings) as sentiment:
                await self._run_cycle(sentiment, allow_revalidation=False)
        finally:
            await self.revalidation.close()
            await asyncio.to_thread(self.execution.shutdown)
            LOGGER.info("MT5 DEMO smoke-test session shut down cleanly")

    async def run(self) -> None:
        if self.settings.predictive_mode == "shadow":
            # Fail before MT5 routing if the active shadow baseline is absent/invalid.
            await asyncio.to_thread(self.quant.load)
        await asyncio.to_thread(self.execution.connect)
        account = self.execution.mt5.account_info()
        if account is None:
            raise RuntimeError("MT5 account unavailable after connection")
        synced = await asyncio.to_thread(self.journal.sync_mt5_history, self.execution.mt5, account)
        LOGGER.info("startup journal sync orders=%d deals=%d", synced["orders"], synced["deals"])
        LOGGER.info(
            "MT5 connected mode=%s routing=%s predictive_mode=%s",
            "DEMO required" if self.settings.require_demo_account else "configured account",
            "ENABLED" if self.settings.trading_enabled and not self.settings.dry_run else "DRY-RUN",
            self.settings.predictive_mode.upper(),
        )
        try:
            async with SentimentEngine(self.settings) as sentiment:
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    try:
                        await self._run_cycle(sentiment)
                    except Exception:
                        LOGGER.exception("execution cycle degraded")
                    delay = max(0.0, self.settings.loop_interval_seconds - (time.monotonic() - started))
                    try:
                        await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass
        finally:
            await self.revalidation.close()
            await asyncio.to_thread(self.execution.shutdown)
            LOGGER.info("MT5 session shut down cleanly")


async def async_main() -> None:
    configure_logging()
    application = TradingApplication()
    loop = asyncio.get_running_loop()
    for event in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(event, application.request_stop)
        except (NotImplementedError, RuntimeError):  # Windows Proactor loop
            pass
    await application.run()


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        LOGGER.info("Keyboard interrupt received")
