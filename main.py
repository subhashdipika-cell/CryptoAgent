"""Asynchronous coordinator for local forecasts, cloud sentiment, and MT5."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from logging.handlers import RotatingFileHandler

from config import LOG_DIR, SETTINGS, Settings
from execution_agent import MT5ExecutionAgent, Side
from quant_engine import ChronosForecastEngine, ForecastResult, close_prices, true_range_atr
from sentiment_engine import SentimentEngine, SentimentResult


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
        self.execution = MT5ExecutionAgent(settings)
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

    async def _process_symbol(self, symbol: str, managed: set[str]) -> tuple[str, float]:
        def load_rates() -> tuple[object, object]:
            # MetaTrader5's Python bridge is treated as single-threaded.
            return (
                self.execution.bars(symbol, self.execution.mt5.TIMEFRAME_M15, self.settings.bar_count),
                self.execution.bars(symbol, self.execution.mt5.TIMEFRAME_H1, self.settings.bar_count),
            )

        m15_rates, h1_rates = await asyncio.to_thread(load_rates)
        atr = true_range_atr(m15_rates, self.settings.atr_period)
        # A single model instance is intentionally called serially; torch pipelines are not assumed thread-safe.
        m15 = await asyncio.to_thread(self.quant.forecast, close_prices(m15_rates), "15min")
        h1 = await asyncio.to_thread(self.quant.forecast, close_prices(h1_rates), "1h")
        side = combined_side(m15, h1, self._sentiment.score, self.settings.signal_threshold)
        LOGGER.info(
            "%s M15=%s/%.3f H1=%s/%.3f ATR=%.5f decision=%s",
            symbol,
            m15.direction,
            m15.probability,
            h1.direction,
            h1.probability,
            atr,
            side.value if side else "HOLD",
        )
        if side and symbol not in managed:
            plan = await asyncio.to_thread(self.execution.build_order, symbol, side, atr)
            await asyncio.to_thread(self.execution.submit, plan)
        return symbol, atr

    async def _run_cycle(self, sentiment: SentimentEngine) -> None:
        await self._refresh_sentiment(sentiment)
        snapshot = await asyncio.to_thread(self.execution.snapshot)
        managed = await asyncio.to_thread(self.execution.managed_position_symbols)
        LOGGER.info(
            "account equity=%.2f free_margin=%.2f positions=%d",
            snapshot.equity,
            snapshot.free_margin,
            snapshot.positions,
        )
        atr_by_symbol: dict[str, float] = {}
        for symbol in self.settings.symbols:
            try:
                name, atr = await self._process_symbol(symbol, managed)
                atr_by_symbol[name] = atr
            except Exception:
                LOGGER.exception("symbol cycle failed for %s", symbol)
        await asyncio.to_thread(self.execution.trail_positions, atr_by_symbol)

    async def run_once(self) -> None:
        """Run one complete dry-run cycle for deployment verification."""
        if self.settings.trading_enabled or not self.settings.dry_run:
            raise PermissionError("run_once is restricted to dry-run mode")
        await asyncio.to_thread(self.quant.load)
        await asyncio.to_thread(self.execution.connect)
        try:
            async with SentimentEngine(self.settings) as sentiment:
                await self._run_cycle(sentiment)
        finally:
            await asyncio.to_thread(self.execution.shutdown)
            LOGGER.info("MT5 DEMO smoke-test session shut down cleanly")

    async def run(self) -> None:
        await asyncio.to_thread(self.quant.load)  # fail before MT5 routing if local model is absent/invalid
        await asyncio.to_thread(self.execution.connect)
        LOGGER.info(
            "MT5 connected mode=%s routing=%s",
            "DEMO required" if self.settings.require_demo_account else "configured account",
            "ENABLED" if self.settings.trading_enabled and not self.settings.dry_run else "DRY-RUN",
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
