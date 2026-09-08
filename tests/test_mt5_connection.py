import unittest
from types import SimpleNamespace as NS
from config import Settings
from mt5_connection import connect_bounded

class Terminal:
    ACCOUNT_TRADE_MODE_DEMO = 0
    def __init__(self, failures=0):
        self.failures, self.calls, self.detaches = failures, 0, 0
        self.connected, self.mode, self.path = True, 0, r'D:\MT5IntelliTrade'
    def shutdown(self): self.detaches += 1
    def initialize(self, **kwargs):
        self.calls += 1
        assert kwargs['timeout'] <= 20000
        return self.calls > self.failures
    def last_error(self): return (-10005, 'IPC timeout')
    def terminal_info(self): return NS(connected=self.connected, path=self.path)
    def account_info(self): return NS(login=1, server='Demo', trade_mode=self.mode, leverage=100)

class RecoveryTests(unittest.TestCase):
    def run_connection(self, terminal):
        return connect_bounded(terminal, Settings(mt5_terminal_path=r'D:\MT5IntelliTrade\terminal64.exe'), sleep=lambda _: None)
    def test_transient_ipc_recovers_with_stable_probes(self):
        t=Terminal(2)
        h=self.run_connection(t)
        self.assertEqual((h['attempts'], h['stable_probes']), (3,3))
        self.assertEqual(len(h['recovery_errors']),2)
    def test_persistent_ipc_is_bounded_and_actionable(self):
        t=Terminal(9)
        with self.assertRaisesRegex(ConnectionError, 'coordinate shared-terminal ownership'): self.run_connection(t)
        self.assertEqual(t.calls,3)
    def test_disconnected_terminal_is_not_ready(self):
        t=Terminal(); t.connected=False
        with self.assertRaises(ConnectionError): self.run_connection(t)
        self.assertEqual(t.calls,3)
    def test_live_and_wrong_terminal_never_retry(self):
        for field,value in [('mode',2),('path',r'C:\other')]:
            t=Terminal(); setattr(t,field,value)
            with self.assertRaises(PermissionError): self.run_connection(t)
            self.assertEqual(t.calls,1)
    def test_discovery_is_forbidden(self):
        with self.assertRaisesRegex(PermissionError,'auto-discovery'):
            connect_bounded(Terminal(), Settings(mt5_terminal_path=''))
