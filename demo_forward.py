"""Bounded DEMO forward evaluation, independent of production policy approval."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from config import BASE_DIR, Settings
from asset_predictive_engine import DedicatedAssetForecastEngine
from decision_engine import CalibratedDecisionEngine
from execution_agent import MT5ExecutionAgent
from quant_engine import true_range_atr

MANIFEST = BASE_DIR / 'policies/demo_forward.json'
STATE = BASE_DIR / 'data/demo_forward.db'


def candidate_hash(path):
    digest = hashlib.sha256(path.read_bytes())
    for name in ('demo_forward.py', 'execution_agent.py', 'decision_engine.py',
                 'asset_predictive_engine.py', 'quant_engine.py'):
        digest.update((BASE_DIR / name).read_bytes())
    return digest.hexdigest()


def validate(manifest, policy_path, now):
    if manifest.get('scope') != 'DEMO_ONLY' or manifest.get('enabled') is not True:
        raise PermissionError('Forward DEMO candidate is not enabled')
    if manifest.get('approved') is not True:
        raise PermissionError('Explicit forward DEMO approval required')
    expiry = datetime.fromisoformat(manifest['expires_at'])
    if expiry.tzinfo is None or now >= expiry or (expiry-now).total_seconds() > 7*86400:
        raise PermissionError('Approval must expire within seven days')
    if manifest.get('configuration_hash') != candidate_hash(policy_path):
        raise PermissionError('Candidate code/policy hash mismatch')
    if not manifest.get('account_login') or not manifest.get('server'):
        raise PermissionError('Exact DEMO account and server required')
    evidence = manifest.get('research_evidence', {})
    if not (evidence.get('untouched') is True and evidence.get('broker_costs_verified') is True
            and evidence.get('stable_walk_forward') is True and evidence.get('trades', 0) >= 30
            and math.isfinite(evidence.get('expectancy', 0)) and evidence.get('expectancy', 0) > 0
            and math.isfinite(evidence.get('profit_factor', 0)) and evidence.get('profit_factor', 0) >= 1.2
            and 0 <= evidence.get('max_drawdown_pct', 100) <= 5):
        raise PermissionError('Untouched broker-aware candidate evidence has not passed')


def reserve(db, key, day):
    """Durable reservation before send: any uncertain send locks future entries."""
    db.execute('BEGIN IMMEDIATE')
    try:
        if db.execute("SELECT 1 FROM attempts WHERE status='RESERVED'").fetchone():
            raise PermissionError('Uncertain prior submission requires reconciliation')
        if db.execute('SELECT count(*) FROM attempts').fetchone()[0] >= 30:
            raise PermissionError('30-attempt evaluation limit reached')
        if db.execute('SELECT count(*) FROM attempts WHERE day=?', (day,)).fetchone()[0] >= 2:
            raise PermissionError('Two-attempt daily limit reached')
        db.execute('INSERT INTO attempts VALUES (?, ?, ?, ?)', (key, day, 'RESERVED', '{}'))
        db.commit()
    except Exception:
        db.rollback()
        raise


def cycle(route=False):
    manifest = json.loads(MANIFEST.read_text())
    candidate = BASE_DIR / 'policies/demo_forward_candidate.json'
    now = datetime.now(timezone.utc)
    # Even dry-run requires a reviewed candidate; never fall back to rejected production policies.
    validate(manifest, candidate, now)
    settings = Settings(symbols=tuple(manifest['symbols']),
        mt5_terminal_path=manifest['terminal_path'], mt5_login=None,
        mt5_password='', mt5_server='', require_demo_account=True,
        trading_enabled=False, dry_run=True, predictive_mode='calibrated',
        strategy_name='DemoForward', magic_number=26091301,
        decision_policy_path=candidate, automatic_revalidation=False)
    settings.validate()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(STATE) as db:
        db.execute('CREATE TABLE IF NOT EXISTS attempts (key TEXT PRIMARY KEY, day TEXT, status TEXT, detail TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS identity (binding TEXT PRIMARY KEY, equity REAL)')
        agent = MT5ExecutionAgent(settings)
        try:
            agent.connect()
            def account_check():
                a = agent.mt5.account_info()
                if a is None or a.trade_mode != 0 or a.login != manifest['account_login'] or a.server != manifest['server']:
                    raise PermissionError('DEMO account identity changed or unavailable')
                return a
            account = account_check()
            binding = json.dumps([manifest['account_login'], manifest['server'], manifest['configuration_hash']])
            saved = db.execute('SELECT binding,equity FROM identity').fetchone()
            if saved and saved[0] != binding:
                raise PermissionError('Journal belongs to another evaluation; archive/review required')
            if not saved:
                db.execute('INSERT INTO identity VALUES (?,?)', (binding, account.equity))
                db.commit()
            initial_equity = saved[1] if saved else account.equity
            positions, orders = agent.mt5.positions_get(), agent.mt5.orders_get()
            deals = agent.mt5.history_deals_get(datetime.fromisoformat(manifest['started_at']), now)
            if positions is None or orders is None or deals is None:
                raise PermissionError('Reconciliation unavailable')
            tagged = [p for p in positions if p.magic == settings.magic_number]
            if tagged or any(o.magic == settings.magic_number for o in orders):
                print('HOLD: an evaluation position/order already exists')
                return
            pnl = sum(d.profit+d.commission+d.swap+getattr(d, 'fee', 0) for d in deals if d.magic == settings.magic_number)
            if pnl <= -20 or initial_equity-account.equity >= 20:
                raise PermissionError('20 account-currency loss limit reached')
            engine = DedicatedAssetForecastEngine(settings)
            decisions = CalibratedDecisionEngine(candidate)
            for symbol in settings.symbols:
                m15 = agent.bars(symbol, agent.mt5.TIMEFRAME_M15, 500)
                h1 = agent.bars(symbol, agent.mt5.TIMEFRAME_H1, 500)
                observation = int(m15[-1]['time'])+900
                if not 0 <= now.timestamp()-observation <= 180:
                    print(f'{symbol}: HOLD stale completed candle')
                    continue
                outcome = decisions.evaluate(symbol, engine.forecast(symbol,m15,'15min'),
                    engine.forecast(symbol,h1,'1h'), .5, True)
                print(f'{symbol}: {outcome.reason}')
                if outcome.side is None:
                    continue
                account = account_check()
                if not math.isfinite(account.equity) or account.equity <= 0:
                    raise PermissionError('Invalid equity')
                # At most 4 currency units planned stop risk, leaving one unit of cost allowance.
                agent.settings = replace(settings, max_risk_fraction=min(.001, 4/account.equity))
                plan = agent.build_order(symbol, outcome.side, true_range_atr(m15))
                tick = agent.mt5.symbol_info_tick(symbol)
                if tick is None or not 0 <= time.time()-tick.time <= 30:
                    raise PermissionError('Fresh tick required')
                if not math.isfinite(plan.risk_amount) or plan.risk_amount > 4:
                    raise PermissionError('Planned risk exceeds fixed allowance')
                if not route:
                    print(f'DRY RUN {symbol}: volume={plan.volume}, stop risk={plan.risk_amount:.2f}')
                    return
                # Check revocation on disk immediately before the atomic reservation/send.
                if json.loads(MANIFEST.read_text()) != manifest:
                    raise PermissionError('Approval changed during evaluation')
                validate(manifest, candidate, datetime.now(timezone.utc))
                account_check()
                bar = int(h1[-1]['time']) if decisions.decision_mode(symbol)=='H1_ONLY' else int(m15[-1]['time'])
                key = f'{symbol}:{bar}'
                reserve(db, key, now.date().isoformat())
                agent.settings = replace(agent.settings, trading_enabled=True, dry_run=False)
                result = agent.submit(plan)
                db.execute('UPDATE attempts SET status=?,detail=? WHERE key=?',
                    ('SENT',json.dumps({'order':result.order,'deal':result.deal,'symbol':symbol}),key))
                db.commit()
                return  # Maximum one entry per cycle across all assets.
        finally:
            agent.shutdown()


def main():
    import msvcrt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route', action='store_true', help='Request explicitly approved DEMO orders')
    p.add_argument('--loop', action='store_true')
    args = p.parse_args()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    # Windows releases this OS lock on process exit, including crashes.
    with (STATE.parent / 'demo_forward.lock').open('a+b') as lock:
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        while True:
            cycle(args.route)
            if not args.loop:
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
