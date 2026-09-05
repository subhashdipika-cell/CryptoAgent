import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
import tempfile
import json
import numpy as np
from config import Settings
from performance_report import sync_from_terminal
from regime_experiment import signals, replay


class RegimeTests(unittest.TestCase):
    def rates(self):
        r = np.zeros(500, dtype=[('time','i8'),('open','f8'),('high','f8'),('low','f8'),('close','f8'),('spread','i4')])
        r['time'] = 1704067200+np.arange(500)*3600
        r['close'] = 100+np.arange(500)*.1
        r['open'] = r['close']-.05
        r['high'] = r['close']+.01
        r['low'] = r['open']-.01
        r['spread'] = 1
        return r

    def test_future_prices_cannot_change_prior_signals(self):
        r = self.rates()
        before = signals(r)
        r['close'][300:] *= 10
        np.testing.assert_array_equal(before[:300], signals(r)[:300])

    def test_session_filters_signal_observation_time(self):
        r = self.rates()
        s = signals(r, session=(7, 16))
        for i in np.flatnonzero(s):
            self.assertIn(((int(r[i]['time'])+3600)//3600)%24, range(7,16))

    def test_minimum_lot_does_not_round_risk_up(self):
        r = self.rates()
        m = dict(point=.01, cash_per_price_lot=100000, volume_min=1, volume_step=1, volume_max=10, stop_distance=0)
        result = replay(r,m,signals(r),100,500)
        self.assertEqual(result['trades'],0)
        self.assertGreater(result['risk_blocked'],0)

    def test_failed_connect_invalidates_previous_success(self):
        with tempfile.TemporaryDirectory() as d:
            settings = replace(Settings(), report_dir=Path(d), journal_path=Path(d)/'j.db')
            marker = Path(d)/'reconciliation_status.json'
            marker.write_text('{"status":"SUCCESS"}')
            with patch('performance_report.MT5ExecutionAgent') as agent:
                agent.return_value.connect.side_effect = ConnectionError('IPC timeout')
                with self.assertRaises(ConnectionError):
                    sync_from_terminal(settings)
            self.assertEqual(json.loads(marker.read_text())['status'],'FAILED')
