"""Isolated, reproducible H1 regime research. Never imports an order router."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def signals(rates, efficiency=0.0, session=(0, 24)):
    """Completed H1 EMA20/EMA50 trend, prior 20-bar breakout, 20-bar efficiency."""
    close = rates['close'].astype(float)
    fast, slow = close[0], close[0]
    result = np.zeros(len(rates), dtype=int)
    for i, price in enumerate(close):
        fast += (price - fast) * 2 / 21
        slow += (price - slow) * 2 / 51
        if i < 100:
            continue
        hour = datetime.fromtimestamp(int(rates[i]['time']) + 3600, timezone.utc).hour
        movement = np.abs(np.diff(close[i-20:i+1])).sum()
        ratio = abs(price-close[i-20]) / movement if movement else 0
        if not session[0] <= hour < session[1] or ratio < efficiency:
            continue
        if price > fast > slow and price > max(rates['high'][i-20:i]):
            result[i] = 1
        elif price < fast < slow and price < min(rates['low'][i-20:i]):
            result[i] = -1
    return result


def replay(rates, metadata, signal, start, end, stress=1.0):
    """Next-bar fills; fixed 2 ATR stop/3 ATR target; exit within six H1 bars.

    A deliberately conservative cash proxy, not MT5 tick replay. Overnight and
    missing hourly bars force a close; no swap-free assumption across rollover.
    """
    equity, peak, dd = 1000.0, 1000.0, 0.0
    trades, blocked = [], 0
    i = max(start, 101)
    while i < end-1:
        side = int(signal[i-1])
        if not side or int(rates[i]['time']) != int(rates[i-1]['time'])+3600:
            i += 1
            continue
        row = rates[i]
        # Avoid holding through an unknown broker rollover; timestamps are UTC.
        if datetime.fromtimestamp(int(row['time']), timezone.utc).hour >= 18:
            i += 1
            continue
        prev = rates[i-15:i]
        atr = float(np.mean(np.maximum(prev['high'][1:]-prev['low'][1:],
            np.maximum(abs(prev['high'][1:]-prev['close'][:-1]),
                       abs(prev['low'][1:]-prev['close'][:-1])))))
        spread = float(row['spread']) * metadata['point'] * stress
        slip = 25 * metadata['point'] * stress
        entry = float(row['open']) + (spread if side == 1 else 0) + side*slip
        distance = 2*atr
        loss_lot = distance*metadata['cash_per_price_lot']
        raw = equity*.005 / (loss_lot + 6*stress + 2*slip*metadata['cash_per_price_lot'])
        step, minimum = metadata['volume_step'], metadata['volume_min']
        volume = math.floor(raw/step)*step
        if distance <= metadata['stop_distance'] or volume < minimum:
            blocked += 1
            i += 1
            continue
        volume = min(volume, metadata['volume_max'])
        stop, target = entry-side*distance, entry+side*3*atr
        j = i
        reason = 'TIME_EXIT'
        while j < min(i+6, end):
            bar = rates[j]
            ask = float(bar['spread'])*metadata['point']*stress if side == -1 else 0
            low, high = float(bar['low'])+ask, float(bar['high'])+ask
            adverse = min(0.0, side*((low if side == 1 else high)-entry)*volume*metadata['cash_per_price_lot'])
            dd = max(dd, peak-(equity+adverse))
            stop_hit = low <= stop if side == 1 else high >= stop
            target_hit = high >= target if side == 1 else low <= target
            if stop_hit:
                opened = float(bar['open'])+ask
                exit_price = min(stop, opened) if side == 1 else max(stop, opened)
                reason = 'AMBIGUOUS_STOP_FIRST' if target_hit else 'STOP'
            elif target_hit:
                exit_price, reason = target, 'TARGET'
            else:
                exit_price = float(bar['close'])+ask
            if stop_hit or target_hit or j+1 >= min(i+6, end) or int(rates[j+1]['time']) != int(bar['time'])+3600:
                break
            j += 1
        net = side*(exit_price-side*slip-entry)*volume*metadata['cash_per_price_lot']-6*stress*volume
        # Track a conservative intratrade adverse excursion as well as closed equity.
        adverse = min(0.0, side*((low if side == 1 else high)-entry)*volume*metadata['cash_per_price_lot'])
        dd = max(dd, peak-(equity+adverse))
        equity += net
        dd = max(dd, peak-equity)
        peak = max(peak, equity)
        trades.append({'entry': int(row['time']), 'exit': int(rates[j]['time']),
                       'net': net, 'volume': volume, 'reason': reason})
        i = j+1
    values = [t['net'] for t in trades]
    wins = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    return {'trades': len(values), 'net': sum(values),
            'expectancy': sum(values)/len(values) if values else 0,
            'pf': wins/losses if losses else (999 if wins else 0),
            'max_drawdown_pct': dd/1000*100,
            'worst_trade': min(values, default=0),
            'largest_win_share': max(values, default=0)/wins if wins else 1,
            'risk_blocked': blocked}


def passes(m):
    return (m['trades'] >= 30 and m['expectancy'] > 0 and m['pf'] >= 1.2
            and m['max_drawdown_pct'] <= 5 and m['largest_win_share'] <= .35)


def experiment(rates, metadata, symbol):
    n = len(rates)
    if n < 2000 or np.any(np.diff(rates['time']) <= 0):
        raise ValueError('Need >=2000 chronological distinct completed H1 bars')
    for field in ('open', 'high', 'low', 'close', 'spread'):
        if not np.all(np.isfinite(rates[field])) or np.any(rates[field] < 0):
            raise ValueError('Invalid OHLC/spread data')
    # Data has been used in earlier research: never relabel this final partition untouched.
    edges = [100, int(n*.4), int(n*.55), int(n*.7), n]
    parameters = ([0.0, .2, .3, .4] if symbol.startswith('BTC')
                  else [(0, 24), (7, 16), (8, 17), (9, 18)])
    batch = []
    for p in parameters:
        s = signals(rates, efficiency=p) if symbol.startswith('BTC') else signals(rates, efficiency=.2, session=p)
        folds = [replay(rates, metadata, s, a, b) for a, b in zip(edges[:3], edges[1:4])]
        batch.append({'parameter': p, 'development': folds})
    # Selection sees development only; final partition evaluated once for the selected candidate.
    selected = max(range(len(batch)), key=lambda k: (
        all(passes(f) for f in batch[k]['development']),
        min(f['expectancy'] for f in batch[k]['development']),
        -max(f['max_drawdown_pct'] for f in batch[k]['development'])))
    p = parameters[selected]
    s = signals(rates, efficiency=p) if symbol.startswith('BTC') else signals(rates, efficiency=.2, session=p)
    final = replay(rates, metadata, s, edges[3], edges[4])
    stressed = replay(rates, metadata, s, edges[3], edges[4], stress=1.5)
    neighbors = [batch[k] for k in (selected-1, selected+1) if 0 <= k < len(batch)]
    stable = all(passes(f) for item in [batch[selected], *neighbors] for f in item['development'])
    return {'symbol': symbol, 'timeframe': 'H1', 'dimension': 'efficiency' if symbol.startswith('BTC') else 'UTC_session',
            'windows': [[int(rates[a]['time']), int(rates[b-1]['time'])] for a,b in zip(edges,edges[1:])],
            'batch': batch, 'selected_parameter': p, 'final_observed_test': final,
            'cost_stress_1_5x': stressed, 'stable_neighbors': stable,
            'metric_gate_pass': stable and passes(final) and passes(stressed),
            'status': 'RESEARCH_ONLY', 'demo_eligible': False,
            'rejection_reason': 'Previously observed history; independent untouched window and verified historical costs required',
            'forward_demo_status': 'NOT_STARTED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true', help='Read completed bars from DEMO MT5')
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; preserve earlier experiments')
    if args.capture:
        if args.snapshot.exists():
            raise FileExistsError('Snapshot is immutable; use a new path')
        import MetaTrader5 as mt5
        import os
        if not mt5.initialize(path=os.environ['MT5_TERMINAL_PATH'], timeout=20000):
            raise RuntimeError(mt5.last_error())
        try:
            account = mt5.account_info()
            if account is None or account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
                raise PermissionError('DEMO account required')
            arrays, meta = {}, {}
            for symbol in ('BTCUSD', 'XAUUSD+'):
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, 6000)
                info = mt5.symbol_info(symbol)
                if rates is None or len(rates) < 2000 or info is None:
                    raise RuntimeError(f'{symbol}: insufficient data/metadata')
                tick = mt5.symbol_info_tick(symbol)
                profit = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, 1.0, tick.ask, tick.ask+1)
                if profit is None or profit <= 0:
                    raise RuntimeError('Profit conversion unavailable')
                arrays[symbol] = rates
                meta[symbol] = dict(point=info.point, volume_min=info.volume_min,
                    volume_max=info.volume_max, volume_step=info.volume_step,
                    stop_distance=info.trade_stops_level*info.point, cash_per_price_lot=profit)
            arrays['metadata'] = np.array(json.dumps(meta))
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.snapshot, **arrays)
        finally:
            mt5.shutdown()
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        meta = json.loads(str(snapshot['metadata']))
        result = [experiment(snapshot[s], meta[s], s) for s in ('BTCUSD', 'XAUUSD+')]
    payload = {'schema_version': 1, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'snapshot_sha256': hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
               'costs': 'Historical bar spread; assumed $3/lot/side commission and 25 points/fill slippage; 1.5x stress. Current broker conversion and lot/stop metadata; no rollover holdings. Historical commission/FX/margin/tick path unverified.',
               'risk': '0.5% proxy risk, $1000 isolated equity per segment; no portfolio claim',
               'routing_changed': False, 'strategies': result}
    payload['configuration_hash'] = fingerprint(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
