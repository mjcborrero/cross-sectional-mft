# Cross-sectional MFT

*A cross-sectional crypto strategy that failed out-of-sample, and the method that found out why.*

This repository is a complete, reproducible research record of a beta-neutral
cross-sectional ranking strategy on 20 Binance perpetual futures at an 8-hour
horizon. It was built under a pre-registered, gated methodology; it passed its
first holdout with a Sharpe of +2.83; it lost money on the second. The record
of *why* it lost is the contribution.

The strategy is not the point. The discipline is. Everything a reader needs to
check every number here is in this repository except the market data itself,
which is downloaded from Binance's public archive by one script.

## The result

| period | role | Sharpe | ann. return | max DD | bars |
|---|---|---|---|---|---|
| 2020-10 → 2024-12 | train (out-of-fold) | +2.86 | +48.7% | −8.6% | 2,755 |
| 2025 | holdout 1 | **+2.83** | +49.4% | −7.3% | 1,095 |
| 2026-01 → 2026-07 | holdout 2 | **−0.35** | −4.2% | −12.4% | 633 |

Fees are assumed zero by design (a maker-only execution model); every figure
above is gross of costs. Both holdouts were spendable exactly once and are
now spent. Nothing in this repository was tuned on either of them.

## Month by month

The full-period line hides what "failed out-of-sample" looked like. 2025 had
three losing months, none worse than −3.4%. 2026 had four, and one of them
was −9.6% — the May drawdown is the violent-tercile mechanism above, in a
single month.

**2025 — holdout 1** (`results/strategy_holdout1.json`)

| month | return | Sharpe | max DD | hit |
|---|---|---|---|---|
| 2025-01 | +1.74% | +1.23 | -3.53% | 48.4% |
| 2025-02 | +8.83% | +7.10 | -4.25% | 56.0% |
| 2025-03 | +2.90% | +2.27 | -4.88% | 54.8% |
| 2025-04 | +2.32% | +2.67 | -2.47% | 54.4% |
| 2025-05 | +7.55% | +6.09 | -1.99% | 59.1% |
| 2025-06 | +7.32% | +11.15 | -1.23% | 70.0% |
| 2025-07 | +5.05% | +3.56 | -2.97% | 51.6% |
| 2025-08 | -1.27% | -1.01 | -5.51% | 58.1% |
| 2025-09 | -1.46% | -1.59 | -3.60% | 45.6% |
| 2025-10 | +10.20% | +6.51 | -2.21% | 55.9% |
| 2025-11 | +1.06% | +0.99 | -3.73% | 51.1% |
| 2025-12 | -3.44% | -3.55 | -5.29% | 47.3% |

9/12 months positive.

**2026 — holdout 2** (`results/strategy_holdout2.json`)

| month | return | Sharpe | max DD | hit |
|---|---|---|---|---|
| 2026-01 | -1.08% | -0.95 | -5.30% | 48.4% |
| 2026-02 | -0.86% | -0.77 | -2.40% | 54.8% |
| 2026-03 | +5.10% | +4.73 | -1.99% | 61.3% |
| 2026-04 | +2.09% | +3.13 | -1.82% | 53.3% |
| 2026-05 | -9.57% | -8.85 | -10.83% | 45.2% |
| 2026-06 | -1.82% | -1.51 | -2.62% | 55.6% |
| 2026-07 | +4.00% | +5.95 | -1.17% | 66.7% |

3/7 months positive.

## Why it failed — the finding

The model's ranking power did not degrade. Its per-bar information coefficient
was +0.041 on train and +0.027 in 2026, and it was positive in every space —
vol-normalised target, dollar residual, raw return. The book lost anyway.

Splitting bars by cross-sectional dispersion of the forward residual
resolves the contradiction:

| dispersion tercile | train Q5−Q1 | 2026 Q5−Q1 |
|---|---|---|
| quiet | +9.11 bps | +9.27 bps |
| mid | +9.17 | +10.27 |
| violent | −14.57 | **−35.52** |
| plain IC | +0.0412 | +0.0274 |
| **dispersion-weighted IC** | +0.0297 | **+0.0007** |

The edge on quiet and mid bars is unchanged out of sample. The entire failure
is the violent tercile. Per-bar IC weights every bar equally; P&L weights each
bar by how much moved. The model is right when little is at stake and wrong
when a lot is, so a positive IC coexists with a losing book — and
`IR ≈ IC·√breadth` never applied.

The effect being captured is short-horizon reversal (the probe IC halves
between a 5-minute and a 60-minute execution lag; nothing survives to 24h or
2–5 days). That is liquidity provision, and liquidity provision has exactly
this payoff shape: small steady wins when the market is orderly, large losses
when a move is real. The 22,000 positions a year that produced the train
Sharpe could not absorb a few hundred of them. Gating the book on forecast
dispersion has the right sign and fails a paired t-test (best cell t = 0.98).

Full account: [`docs/PROJECT_FINDINGS.md`](docs/PROJECT_FINDINGS.md), §11.

## The discipline

These are the rules the project ran under. Each one is enforced in code, not
in a document.

**Pre-registration.** Every threshold, gate and decision rule was written down
before the data that would test it was looked at. Where a bar was later moved,
it is recorded as post-hoc in the design document, not presented as design.
[`docs/TARGET_DESIGN.md`](docs/TARGET_DESIGN.md),
[`docs/FEATURE_SELECTION.md`](docs/FEATURE_SELECTION.md).

**Holdouts are spendable once.** Two walls, 2025-01-01 and 2026-01-01.
`mft/splits.py` refuses reads past the active wall, and `run_strategy.py`
refuses to evaluate a holdout unless `--i-am-burning-a-holdout` is passed —
an accidental invocation is not recoverable, so it has to be deliberate.

**Gates abort; they do not print.** The build chain (decision grid → weights →
index → betas → forward legs → target) is eight gated steps. A failed gate
writes nothing. Gate 8B rebuilds the target at sampled instants from raw
1-minute klines through an independent code path and requires agreement to
1e-6 — because re-running the same code and getting the same answer proves
nothing. `scripts/build_target.py`.

**Point-in-time, asserted at every layer.** `scripts/audit_data.py` runs ~30
checks from raw klines to the assembled target: no observation price uses a
close stamped after `t_obs`, betas at `t` reproduce an independent rolling
OLS using only data at or before `t`, fill prices are the close as-of
`t_obs + lag` and never later, weights are built from volume observed before
the bar. Each is recomputed from a different code path and compared, not
re-read from the artifact being checked.

**Multiple-testing correction over mechanism families, not columns.** 72
features were frozen from a larger candidate set through seven staged filters
with pre-registered pass criteria; the correction counted the families a
feature could belong to, not the columns tested.
[`docs/FEATURE_EVALUATION.md`](docs/FEATURE_EVALUATION.md),
[`docs/FEATURE_LIST_FROZEN.md`](docs/FEATURE_LIST_FROZEN.md).

**Seed ensembling.** A single LightGBM seed is an arbitrary draw (measured
Sharpe range 2.51–3.04 across 12 seeds). Eight fixed seeds are averaged and
the list is declared, so it is not a tuning knob. `scripts/stage7_importance.py`.

**The number must make sense.** Nine falsification checks run against every
result: placebo (shuffled target), P&L reconstruction from weights, target
identity, book invariants, purge, seal, concentration, fee ceiling,
dose-response. `scripts/audit.py`. Most of the bugs listed below were found
because a number was implausible, not because a test failed.

**Parameter sweeps are read for plateaus, not best cells.** A result that
holds at one parameter value and not at its neighbours is noise. This killed
a 30-day momentum lookback that looked good in isolation, and it is why every
sweep in this repository prints the whole grid.

**Both spent holdouts are diagnostic, not evidence.** Everything run after
2026 was read is labelled as such in code and in the findings.

## What went wrong along the way

[`docs/PROJECT_FINDINGS.md`](docs/PROJECT_FINDINGS.md) §6 lists every defect
found. The pattern worth knowing: parameters expressed in **bar counts**
silently change meaning when the clock changes (three instances); a stale
**reference** inside a verifier makes the verifier wrong, not the build (two
instances, both caught by a gate failing at correlation ~0.85); and a
**forward quantity** used at decision time reads as a spectacular signal
(one instance, IC +0.115, entirely spurious). The two verifier bugs were only
found because the gates were strict enough to fail on a good build.

## Repository layout

```
mft/                 the library
  splits.py            holdout walls and the seal
  folds.py             purged expanding walk-forward folds
  strategy.py          the frozen book: weights, smoothing, hedge overlay, metrics
  featbuild.py         feature build harness (point-in-time)
  featdefs/            the 9 feature families that produce the 72 frozen features
  rolling.py           rolling beta / idiosyncratic vol
  data/                Binance public-archive client, manifest, parquet store
scripts/             the pipeline and the research, in build order
  backfill.py          download source data
  build_*.py           the gated build chain
  stage2..8_*.py       feature evaluation stages
  run_strategy.py      execute the frozen spec on a split
  audit*.py            falsification checks
  ridge.py, ic_*.py, dispersion_*.py, rank_hits.py, ...   the post-mortem research
docs/                design documents (pre-registered) and findings
results/             every result JSON the findings cite, so numbers can be checked
config/              the 20-coin universe (the frozen feature list is results/frozen_list.json)
tests/
```

## Reproducing

```bash
pip install -r requirements.txt
export MFT_DATA_DIR=/path/with/space          # roughly 15-20 GB of source data

python scripts/backfill.py --symbols universe    # Binance public archive
python scripts/build_returns_1h.py
python scripts/build_decision_grid.py
python scripts/build_market_weights.py
python scripts/build_market_index.py
python scripts/build_beta_idiovol.py
python scripts/build_forward_returns.py
python scripts/build_target.py                   # Gate 8 aborts on any mismatch
python scripts/build_features.py
python scripts/stage2_estimation.py  ...  stage8_selection.py
python scripts/freeze_list.py
python scripts/run_strategy.py --split train
python scripts/audit.py
```

The holdout splits exist to be spent once. If you reproduce them you are
reading data this project already read; the numbers will match `results/`,
and they will not be out-of-sample for you either.

## What is not here

- **Market data.** Downloaded, not committed. `scripts/backfill.py`.
- **The live execution system.** Out of scope for a research record.
- **The broader feature exploration.** Only the code that produces the 72
  frozen features is included, together with the documents recording how
  they were selected from the candidates.

## License

MIT — see [`LICENSE`](LICENSE).
