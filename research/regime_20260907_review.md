# Regime experiment review — 7 September 2026

Implementation baseline: 2888177. Full experiment configuration, source and
snapshot hashes, chronological windows, development folds, selected parameters,
and 1.5x cost stress are recorded in regime_20260907.json.

BTC selected efficiency 0.3: final observed partition 3 trades, net -1.3057,
expectancy -0.4352, PF 0.6451, proxy drawdown 0.6296%; 32 minimum-lot risk
blocks. Stress: net -1.5714, PF 0.5764. Neighbor stability failed.

XAU selected session 00–24 UTC (subject to the fixed pre-18 UTC entry rule):
0 trades and 98 minimum-lot risk blocks. Neighbor stability failed. Zero trades
does not demonstrate safety or profitability.

Both candidates rejected for further DEMO evaluation. Historical cost conversion,
commission and intrabar execution remain proxies; the final partition overlaps
previously observed history. No untouched performance or future profitability is
claimed. No policy approval, router start, terminal restart, or trade was performed.

Connectivity: DEMO reconciliation and a separate H1 capture succeeded against the
existing D-drive terminal without restarting it. This establishes current access,
not permanent resolution of the intermittent IPC failure.

Repair: recheck current reconciliation and registry before every application
submission; forbid shadow mode from reusing calibrated deployment approval.
Existing position management remains independent of entry-readiness checks.

Next prerequisite: establish broker minimum-lot feasibility and verified costs
before selecting a different holding horizon. Do not increase risk simply to
obtain the required trade count. New model selection needs a fresh future test.
