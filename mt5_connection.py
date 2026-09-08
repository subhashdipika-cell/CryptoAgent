"""Bounded MT5 attachment recovery. Never restarts terminals or sends orders."""
import ntpath
import time


def inspect_connection(mt5, settings):
    terminal, account = mt5.terminal_info(), mt5.account_info()
    if terminal is None or not terminal.connected or account is None:
        raise ConnectionError(f"MT5 broker/account unavailable: {mt5.last_error()}")
    expected = ntpath.normcase(ntpath.normpath(ntpath.dirname(settings.mt5_terminal_path)))
    actual = ntpath.normcase(ntpath.normpath(getattr(terminal, 'path', '')))
    if expected != actual:
        raise PermissionError(f"MT5 terminal ownership mismatch: expected {expected}, attached {actual}")
    if settings.require_demo_account and account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        raise PermissionError('DEMO account required; attachment rejected')
    if settings.mt5_login and account.login != settings.mt5_login:
        raise PermissionError('MT5 account identity mismatch')
    if settings.mt5_server and account.server != settings.mt5_server:
        raise PermissionError('MT5 server identity mismatch')
    if not settings.min_leverage <= account.leverage <= settings.max_leverage:
        raise PermissionError('MT5 leverage outside configured limits')
    return account, {'connected': True, 'terminal_path': terminal.path,
                     'server': account.server, 'trade_mode': account.trade_mode}


def connect_bounded(mt5, settings, *, sleep=time.sleep, clock=time.monotonic):
    if not settings.mt5_terminal_path:
        raise PermissionError('Set MT5_TERMINAL_PATH to the owned terminal executable; auto-discovery is disabled')
    started = clock()
    failures = []
    kwargs = {'path': settings.mt5_terminal_path, 'timeout': min(20000, max(1, settings.mt5_timeout_ms))}
    if settings.mt5_login:
        kwargs.update(login=settings.mt5_login, password=settings.mt5_password, server=settings.mt5_server)
    for attempt in range(1, 4):
        mt5.shutdown()  # detach this Python client only
        try:
            if not mt5.initialize(**kwargs):
                raise ConnectionError(f'MT5 initialize failed: {mt5.last_error()}')
            account, health = inspect_connection(mt5, settings)
            identity = (account.login, account.server, account.trade_mode)
            for _ in range(2):
                sleep(1)
                account, health = inspect_connection(mt5, settings)
                if (account.login, account.server, account.trade_mode) != identity:
                    raise PermissionError('MT5 account changed during health validation')
            return {**health, 'attempts': attempt, 'stable_probes': 3,
                    'elapsed_seconds': round(clock()-started, 3), 'recovery_errors': failures}
        except PermissionError:
            mt5.shutdown()
            raise
        except ConnectionError as error:
            failures.append(str(error))
            mt5.shutdown()
            if attempt < 3 and clock()-started < 65:
                sleep(1)
            else:
                break
    raise ConnectionError(f'MT5 attachment exhausted {len(failures)} attempts: {failures}. '
                          f'Expected executable: {settings.mt5_terminal_path}. '
                          'Verify this terminal is signed into DEMO and connected, and coordinate shared-terminal ownership. '
                          'No terminal was restarted. Native initialize timeout is capped at 20000 ms per attempt; '
                          'other native API calls are not hard wall-clock bounded.')
