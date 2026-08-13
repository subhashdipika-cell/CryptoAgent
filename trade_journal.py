"""Durable SQLite journal and idempotent MetaTrader 5 history reconciliation."""

from __future__ import annotations

import sqlite3
import threading
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from config import Settings
from execution_agent import AccountSnapshot, OrderPlan


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS account_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    account_login INTEGER NOT NULL,
    server TEXT NOT NULL,
    trade_mode INTEGER,
    balance REAL,
    equity REAL NOT NULL,
    margin REAL NOT NULL,
    margin_free REAL NOT NULL,
    leverage INTEGER NOT NULL,
    open_positions INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    account_login INTEGER NOT NULL,
    server TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    m15_direction TEXT NOT NULL,
    m15_probability REAL NOT NULL,
    h1_direction TEXT NOT NULL,
    h1_probability REAL NOT NULL,
    sentiment REAL NOT NULL,
    sentiment_degraded INTEGER NOT NULL,
    atr REAL NOT NULL,
    decision TEXT NOT NULL,
    equity REAL NOT NULL,
    free_margin REAL NOT NULL,
    expert_id INTEGER NOT NULL,
    order_comment TEXT NOT NULL
    ,decision_reason TEXT NOT NULL DEFAULT 'LEGACY_UNKNOWN'
    ,calibrated_score REAL
    ,required_score REAL
    ,active_model TEXT
);

CREATE TABLE IF NOT EXISTS model_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    account_login INTEGER NOT NULL,
    server TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    model_name TEXT NOT NULL,
    mode TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    edge_bps REAL NOT NULL,
    predictions_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submitted_at TEXT NOT NULL,
    account_login INTEGER NOT NULL,
    server TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    strategy TEXT NOT NULL,
    requested_volume REAL NOT NULL,
    requested_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    atr REAL NOT NULL,
    estimated_risk REAL NOT NULL,
    expert_id INTEGER NOT NULL,
    order_comment TEXT NOT NULL,
    order_ticket INTEGER,
    deal_ticket INTEGER,
    executed_volume REAL,
    executed_price REAL,
    retcode INTEGER,
    broker_message TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS mt5_orders (
    ticket INTEGER PRIMARY KEY,
    time_msc INTEGER NOT NULL,
    position_id INTEGER,
    account_login INTEGER NOT NULL,
    server TEXT NOT NULL,
    symbol TEXT NOT NULL,
    type INTEGER,
    state INTEGER,
    volume_initial REAL,
    volume_current REAL,
    price_open REAL,
    stop_loss REAL,
    take_profit REAL,
    price_current REAL,
    expert_id INTEGER NOT NULL,
    comment TEXT,
    reason INTEGER,
    external_id TEXT,
    synced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mt5_deals (
    ticket INTEGER PRIMARY KEY,
    order_ticket INTEGER,
    position_id INTEGER NOT NULL,
    time_msc INTEGER NOT NULL,
    account_login INTEGER NOT NULL,
    server TEXT NOT NULL,
    symbol TEXT NOT NULL,
    type INTEGER,
    entry INTEGER,
    volume REAL,
    price REAL,
    profit REAL NOT NULL DEFAULT 0,
    commission REAL NOT NULL DEFAULT 0,
    swap REAL NOT NULL DEFAULT 0,
    fee REAL NOT NULL DEFAULT 0,
    expert_id INTEGER NOT NULL,
    comment TEXT,
    reason INTEGER,
    external_id TEXT,
    synced_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals(symbol, recorded_at);
CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_time ON model_forecasts(symbol, recorded_at);
CREATE INDEX IF NOT EXISTS idx_deals_position ON mt5_deals(position_id, time_msc);
CREATE INDEX IF NOT EXISTS idx_deals_magic_time ON mt5_deals(expert_id, time_msc);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _value(record: Any, name: str, default: Any = None) -> Any:
    return getattr(record, name, default)


class TradeJournal:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = Path(settings.journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate_schema(connection)

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(signals)")}
        additions = {
            "decision_reason": "TEXT NOT NULL DEFAULT 'LEGACY_UNKNOWN'",
            "calibrated_score": "REAL",
            "required_score": "REAL",
            "active_model": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE signals ADD COLUMN {name} {definition}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_account(self, account: Any, snapshot: AccountSnapshot) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO account_snapshots
                (recorded_at, account_login, server, trade_mode, balance, equity,
                 margin, margin_free, leverage, open_positions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    utc_now(),
                    int(account.login),
                    str(account.server),
                    _value(account, "trade_mode"),
                    float(_value(account, "balance", 0.0)),
                    snapshot.equity,
                    snapshot.margin,
                    snapshot.free_margin,
                    snapshot.leverage,
                    snapshot.positions,
                ),
            )

    def record_model_forecast(
        self, account: Any, symbol: str, timeframe: str, forecast: Any, mode: str
    ) -> None:
        values = [float(value) for value in forecast.predictions]
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO model_forecasts
                (recorded_at, account_login, server, symbol, timeframe, model_name,
                 mode, direction, confidence, edge_bps, predictions_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    utc_now(), int(account.login), str(account.server), symbol, timeframe,
                    forecast.model_name, mode, forecast.direction, float(forecast.probability),
                    float(forecast.edge_bps), json.dumps(values),
                ),
            )

    def record_signal(
        self,
        *,
        account: Any,
        snapshot: AccountSnapshot,
        symbol: str,
        strategy: str,
        m15: Any,
        h1: Any,
        sentiment: Any,
        atr: float,
        decision: str,
        decision_reason: str = "LEGACY_UNKNOWN",
        calibrated_score: float | None = None,
        required_score: float | None = None,
        active_model: str | None = None,
    ) -> None:
        comment = self.settings.order_comment(strategy)
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO signals
                (recorded_at, account_login, server, symbol, strategy,
                 m15_direction, m15_probability, h1_direction, h1_probability,
                 sentiment, sentiment_degraded, atr, decision, equity, free_margin,
                 expert_id, order_comment, decision_reason, calibrated_score,
                 required_score, active_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    utc_now(),
                    int(account.login),
                    str(account.server),
                    symbol,
                    strategy,
                    m15.direction,
                    float(m15.probability),
                    h1.direction,
                    float(h1.probability),
                    float(sentiment.score),
                    int(sentiment.degraded),
                    float(atr),
                    decision,
                    snapshot.equity,
                    snapshot.free_margin,
                    self.settings.magic_number,
                    comment,
                    decision_reason,
                    calibrated_score,
                    required_score,
                    active_model,
                ),
            )

    def record_submission(
        self,
        account: Any,
        plan: OrderPlan,
        result: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO submissions
                (submitted_at, account_login, server, symbol, side, strategy,
                 requested_volume, requested_price, stop_loss, take_profit, atr,
                 estimated_risk, expert_id, order_comment, order_ticket, deal_ticket,
                 executed_volume, executed_price, retcode, broker_message, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    utc_now(),
                    int(account.login),
                    str(account.server),
                    plan.symbol,
                    plan.side.value,
                    plan.strategy_name,
                    plan.volume,
                    plan.entry,
                    plan.stop_loss,
                    plan.take_profit,
                    plan.atr,
                    plan.risk_amount,
                    self.settings.magic_number,
                    self.settings.order_comment(plan.strategy_name),
                    _value(result, "order"),
                    _value(result, "deal"),
                    _value(result, "volume"),
                    _value(result, "price"),
                    _value(result, "retcode"),
                    _value(result, "comment"),
                    str(error) if error else None,
                ),
            )

    def _is_tracked(self, record: Any) -> bool:
        magic = int(_value(record, "magic", 0) or 0)
        comment = str(_value(record, "comment", "") or "")
        return magic in self.settings.tracked_magic_numbers or comment.startswith(
            ("CryptoAgent", "placed by CryptoAgent")
        )

    def sync_mt5_history(self, mt5: Any, account: Any) -> dict[str, int]:
        # Some brokers expose history timestamps in server time rather than host UTC.
        # A small future boundary includes the current server day without fabricating records.
        end = datetime.now(timezone.utc) + timedelta(days=3)
        start = end - timedelta(days=self.settings.history_sync_days)
        with self._connect() as connection:
            latest = connection.execute(
                """SELECT MAX(time_msc) FROM (
                SELECT MAX(time_msc) AS time_msc FROM mt5_orders
                UNION ALL SELECT MAX(time_msc) FROM mt5_deals)"""
            ).fetchone()[0]
        if latest:
            overlap = datetime.fromtimestamp(int(latest) / 1000, timezone.utc) - timedelta(days=2)
            start = max(start, overlap)
        orders = mt5.history_orders_get(start, end)
        deals = mt5.history_deals_get(start, end)
        if orders is None or deals is None:
            raise RuntimeError(f"MT5 history reconciliation failed: {mt5.last_error()}")
        tracked_orders = [record for record in orders if self._is_tracked(record)]
        tracked_deals = [record for record in deals if self._is_tracked(record)]
        synced_at = utc_now()
        with self._lock, self._connect() as connection:
            for order in tracked_orders:
                connection.execute(
                    """INSERT INTO mt5_orders
                    (ticket, time_msc, position_id, account_login, server, symbol,
                     type, state, volume_initial, volume_current, price_open, stop_loss,
                     take_profit, price_current, expert_id, comment, reason, external_id, synced_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticket) DO UPDATE SET
                     state=excluded.state, volume_current=excluded.volume_current,
                     price_current=excluded.price_current, comment=excluded.comment,
                     synced_at=excluded.synced_at""",
                    (
                        int(order.ticket),
                        int(_value(order, "time_msc", int(_value(order, "time", 0)) * 1000)),
                        int(_value(order, "position_id", 0) or 0),
                        int(account.login),
                        str(account.server),
                        str(_value(order, "symbol", "")),
                        _value(order, "type"),
                        _value(order, "state"),
                        float(_value(order, "volume_initial", 0.0)),
                        float(_value(order, "volume_current", 0.0)),
                        float(_value(order, "price_open", 0.0)),
                        float(_value(order, "sl", 0.0)),
                        float(_value(order, "tp", 0.0)),
                        float(_value(order, "price_current", 0.0)),
                        int(_value(order, "magic", 0) or 0),
                        str(_value(order, "comment", "")),
                        _value(order, "reason"),
                        str(_value(order, "external_id", "")),
                        synced_at,
                    ),
                )
            for deal in tracked_deals:
                connection.execute(
                    """INSERT INTO mt5_deals
                    (ticket, order_ticket, position_id, time_msc, account_login, server,
                     symbol, type, entry, volume, price, profit, commission, swap, fee,
                     expert_id, comment, reason, external_id, synced_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticket) DO UPDATE SET
                     profit=excluded.profit, commission=excluded.commission,
                     swap=excluded.swap, fee=excluded.fee, comment=excluded.comment,
                     synced_at=excluded.synced_at""",
                    (
                        int(deal.ticket),
                        int(_value(deal, "order", 0) or 0),
                        int(_value(deal, "position_id", 0) or 0),
                        int(_value(deal, "time_msc", int(_value(deal, "time", 0)) * 1000)),
                        int(account.login),
                        str(account.server),
                        str(_value(deal, "symbol", "")),
                        _value(deal, "type"),
                        _value(deal, "entry"),
                        float(_value(deal, "volume", 0.0)),
                        float(_value(deal, "price", 0.0)),
                        float(_value(deal, "profit", 0.0)),
                        float(_value(deal, "commission", 0.0)),
                        float(_value(deal, "swap", 0.0)),
                        float(_value(deal, "fee", 0.0)),
                        int(_value(deal, "magic", 0) or 0),
                        str(_value(deal, "comment", "")),
                        _value(deal, "reason"),
                        str(_value(deal, "external_id", "")),
                        synced_at,
                    ),
                )
        return {"orders": len(tracked_orders), "deals": len(tracked_deals)}

    def rows(self, table: str) -> list[sqlite3.Row]:
        allowed = {
            "account_snapshots", "signals", "model_forecasts", "submissions",
            "mt5_orders", "mt5_deals",
        }
        if table not in allowed:
            raise ValueError(f"unsupported journal table: {table}")
        with self._connect() as connection:
            return list(connection.execute(f"SELECT * FROM {table}"))
