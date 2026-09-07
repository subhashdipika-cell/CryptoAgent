import sqlite3
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from demo_forward import reserve, validate, cycle


class ForwardTests(unittest.TestCase):
    def db(self, path=':memory:'):
        db = sqlite3.connect(path)
        db.execute('CREATE TABLE IF NOT EXISTS attempts (key TEXT PRIMARY KEY, day TEXT, status TEXT, detail TEXT)')
        self.addCleanup(db.close)
        return db

    def test_unknown_submission_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            first = self.db(str(Path(d)/'test.db'))
            reserve(first,'BTC:1','2026-09-07')
            second = self.db(str(Path(d)/'test.db'))
            with self.assertRaisesRegex(PermissionError,'Uncertain'):
                reserve(second,'XAU:2','2026-09-08')
            second.close()
            first.close()

    def test_daily_limit_and_duplicate_bar(self):
        db = self.db()
        reserve(db,'BTC:1','2026-09-07')
        db.execute("UPDATE attempts SET status='SENT'")
        db.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            reserve(db,'BTC:1','2026-09-07')
        reserve(db,'BTC:2','2026-09-07')
        db.execute("UPDATE attempts SET status='SENT'")
        db.commit()
        with self.assertRaisesRegex(PermissionError,'daily'):
            reserve(db,'BTC:3','2026-09-07')

    def test_disabled_mode_never_constructs_broker(self):
        with patch('demo_forward.MT5ExecutionAgent') as broker:
            with self.assertRaises(PermissionError):
                cycle(route=True)
            broker.assert_not_called()

    def test_non_demo_scope_rejected_before_file_reads(self):
        with self.assertRaises(PermissionError):
            validate({'scope':'LIVE','enabled':True},Path('missing'),datetime.now(timezone.utc))

    def test_total_limit(self):
        db = self.db()
        for i in range(30):
            db.execute('INSERT INTO attempts VALUES (?,?,?,?)',(str(i),str(i),'SENT','{}'))
        db.commit()
        with self.assertRaisesRegex(PermissionError,'30-attempt'):
            reserve(db,'new','new-day')

    def test_approval_requires_research_and_matching_code(self):
        now = datetime.now(timezone.utc)
        manifest = dict(scope='DEMO_ONLY',enabled=True,approved=True,
            expires_at=(now+timedelta(days=1)).isoformat(), configuration_hash='hash',
            account_login=123,server='Broker-Demo',research_evidence={})
        with patch('demo_forward.candidate_hash',return_value='hash'):
            with self.assertRaisesRegex(PermissionError,'evidence'):
                validate(manifest,Path('candidate'),now)
            manifest['research_evidence'] = dict(untouched=True,broker_costs_verified=True,
                stable_walk_forward=True,trades=30,expectancy=1,profit_factor=1.3,max_drawdown_pct=3)
            validate(manifest,Path('candidate'),now)
            manifest['configuration_hash'] = 'changed'
            with self.assertRaisesRegex(PermissionError,'hash'):
                validate(manifest,Path('candidate'),now)
