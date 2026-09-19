# Feature selection — the target-touching plan

**Status: pre-registered, not yet run. Written before any feature has been
compared to the target.**

Everything in `FEATURE_EVALUATION.md` was target-free and therefore unlimited.
Everything here is not. The list was frozen first, deliberately, so that the
choices below are recorded rather than discovered.

| | |
|---|---|
| frozen candidates | **72** (`frozen_list.json`, hash `bd22067e3d6325e0`) |
| mechanism families | **9** (A–I) |
| train bars | **4,680** |
| median coins per bar | **19** |
| panel rows | 90,023 |

---

## 1. The sample is smaller than it looks

90,023 rows is not the sample size. For a **ranking** objective the independent
unit is the **bar**: the 19 coins inside one cross-section are ranked against
each other, so they are one joint observation, not nineteen.

The real ratio is **72 features against 4,680 observations — about 65 bars per
feature.** That is thin. It is the single most important constraint on
everything below, and it rules out entire method classes on its own (§6).

**One thing the target design gets right.** `t_fill = t_obs + 60min` and the
window is `[t_fill, t_fill + 8h]`, so bar *t* spans `[t+1h, t+9h]` and bar *t+1*
spans `[t+9h, t+17h]` — **adjacent and non-overlapping**. There is no
overlap-induced autocorrelation in the target and no need for the
Newey–West-style corrections most horizon-labelled studies require. The purge
between train and holdout still applies (`h + lag`); it is the *within-train*
overlap that is absent.

---

## 2. The scaffold, which matters more than the method

> **Selection happens INSIDE each fold, never once on all of train.**

Selecting on full train and then cross-validating the selected set is the most
common way feature selection lies, and it is the exact failure class the
post-mortem documents. A feature chosen using fold *k*'s data cannot then be
evaluated on fold *k*.

- **Purged, embargoed walk-forward**, inside TRAIN only
- Purge `h + lag` = 9h at every fold boundary, matching `mft/splits.py`
- 2025 (holdout-1) and 2026 (holdout-2) are not touched by anything in this
  document

Everything below is computed per fold and aggregated across folds.

---

## 3. Stage 6 — univariate screen, deliberately lenient

`FEATURE_EVALUATION.md` §8 pre-registered the rule: **"remove features that are
provably nothing, not pick winners."**

Selecting winners univariately does two bad things — it overfits, and it
deletes features that only work in combination. `crowd_winning`
(`FEATURE_EXPLORATION.md` §1.1) is expected to be flat on its own and to matter
interacted; a selective univariate screen would delete it for being exactly
what it was designed to be.

**Metric.** Per-bar Spearman IC between feature and target, then the
distribution of those per-bar ICs. Because bars are non-overlapping (§1), the
t-statistic on that distribution is honest without adjustment.

**The bar is near-zero-with-tight-error, not "below median."** A feature is
removed only when its IC is indistinguishable from zero *and* the error is
small enough that a real effect would have shown. Wide error bars mean "not
measured", which is not the same as "nothing", and must not be treated as such.

Context columns are exempt: they have no cross-sectional ordering, so a
cross-sectional IC is undefined for them. They enter only as interactions.

---

## 4. Stage 7 — feature importance

### The problem with the usual answers

| method | measures | why it misleads here |
|---|---|---|
| Gain / split count | how much the model used the column | biased toward continuous and high-cardinality features; unreliable under correlation |
| SHAP | per-row attribution | correlated features split credit arbitrarily; the plots look far more authoritative than the numbers are |
| Plain permutation | metric drop when a column is shuffled | permute one of a correlated pair and the model leans on the other — **both look unimportant** |

The failure is shared and it is not academic: it is the reason a feature can
rank last on importance and still be doing real work.

### What is actually run

**Grouped permutation (MDA) inside purged folds.** Correlated features are
permuted *together*, so credit cannot leak to a twin.

**Permutation is within-bar.** Shuffling globally destroys the cross-sectional
structure the model ranks on and would measure the wrong thing — it would
mostly detect that the feature has a time trend.

**Grouping, measured on the frozen 72:**

| threshold | pairs |
|---|---|
| ≥ 0.50 | 27 |
| ≥ 0.60 | 11 |
| ≥ 0.70 | 5 |
| ≥ 0.80 | 2 |

Connected components at 0.70 give **67 groups from 72 features** — only five
groups have more than one member:

```
bar_close_location    ~ close_vs_vwap
basis_acceleration    ~ perp_spot_return_gap
coin_sector_corr      ~ sector_cohesion
hurst_change          ~ hurst_short_minus_long
resid_reversal_8h     ~ sector_rel_reversal_8h
```

**This is a small correction, and saying so is the point.** Stage 4 already
removed the pairwise-redundant and the multivariate-explained, so the
correlation problem that usually wrecks importance rankings has largely been
handled upstream. Grouped permutation is still the right default — it costs
almost nothing here and it is honest about the five cases that remain — but it
should not be presented as the thing that makes the analysis work.

### The bar: shadow features

An importance number means nothing without a null. For each fold, add
**shadow columns** — copies of real features shuffled within bar, carrying the
same marginal distribution and no signal. A real feature must beat the
**maximum** shadow importance, not the mean.

Taking the max is what makes this multiple-testing-aware: with *n* shadows the
max is the order statistic a genuinely null feature would have to clear.

---

## 5. Stage 8 — selection by stability, not by score

The final criterion is **not** mean importance across folds. It is the
**fraction of folds in which a feature clears its shadow bar**.

At 65 bars per feature, a single fold's ranking is noise. A feature that wins
one fold and loses four is a fluctuation; one that clears the bar in most folds
is a finding. Stability selection targets false discoveries directly, which is
the error we care about.

**Multiple-testing correction is applied over the 9 mechanism families, not
over the 72 columns** — the project's existing rule. Four lookbacks of one idea
are one hypothesis, not four, and correcting per column would both overcorrect
and misattribute.

---

## 6. Considered and rejected, with reasons

| method | why not |
|---|---|
| **RFE / forward / backward selection** | greedy iteration at 65 bars per feature reliably finds structure in noise |
| **Raw SHAP ranking** | §4; and it invites reading the plot as a conclusion |
| **LightGBM gain as the criterion** | §4; fine as a diagnostic, not as a decision rule |
| **Selecting on full train, CV afterwards** | §2 — this is the failure mode, not a shortcut around it |
| **Knockoffs** (Barber–Candès) | the most rigorous option available and the only one with provable FDR control. Not run because constructing valid knockoffs for dependent, cross-sectional financial data is a research problem in itself, and getting it subtly wrong gives false confidence rather than an error. Recorded as the principled alternative, not dismissed. |

---

## 7. Pre-registered expectations

Written down now so they cannot be rationalised after the numbers arrive.

1. **Expect 10–25 features to survive, not 72.** At 65 bars per feature, a set
   that stays large is more likely to indicate a leaky screen than a rich
   signal.
2. **Expect most univariate ICs to be small.** Cross-sectional crypto ICs in
   the 0.01–0.04 range are normal. An IC above ~0.10 on 8h horizons should be
   treated as a suspected bug until it survives a leakage check, not as a
   discovery.
3. **A sign check costs nothing and is hard to fool.** Families A–G carry
   stated priors. A feature whose realised IC is *opposite* to its
   pre-registered sign is evidence of noise, not of a discovery — and it must
   be reported as such rather than reinterpreted with a new story.
4. **The set must beat the benchmark, not zero.** `resid_reversal_8h` alone is
   the reference point (`FEATURE_LIST_FROZEN.md` §6 decision rule 1). This is
   why the benchmark was exempted from the freeze's multivariate pass.

---

## 8. Settled before Stage 6, and what is still open

### Folds — FIXED, `mft/folds.py`

5 purged expanding walk-forward folds inside TRAIN. Min train 1,800 bars
(~600 days), test 576 bars (192 days) each, tiling the remaining 2,880 exactly.
Purge 2 bars = `ceil((h + lag) / 8h)` = `ceil(9h / 8h)`.

| fold | test period |
|---|---|
| 0 | 2022-05-17 → 2022-11-24 |
| 1 | 2022-11-25 → 2023-06-04 |
| 2 | 2023-06-05 → 2023-12-13 |
| 3 | 2023-12-14 → 2024-06-22 |
| 4 | 2024-06-23 → 2024-12-31 |

Expanding rather than rolling: at 65 bars per feature, discarding history is
the more expensive error. The blocks span both the 2022 bear and the 2023–24
recovery, so "clears the bar in most folds" is a statement about regimes and
not only about time.

### Semi-beta — SETTLED, and it FAILED

`scripts/gate_semibeta.py`, full record in `TARGET_DESIGN.md` §9 item 8.

The target is **linearly** neutral (`corr(target, r_m)` = −0.0039, t −1.15) but
carries **convexity**: `corr(target, |r_m|)` = −0.0435, t −12.91. Splitting by
the sign of `r_m` made this look like asymmetry; it is not.

**Consequence for this document.** The common convexity cannot reach a
cross-sectional ranking objective. The **dispersion** can: per-coin convexity
has std 0.0440 across the 20 coins. So a ninth pre-registered expectation:

5. **The final book must be checked for realised `|r_m|` exposure**, not only
   for `r_m` exposure. A book that sorts on any characteristic correlated with
   coin-level convexity inherits it, and no stage before the backtest can see
   that. This is registered now so it is not discovered as a surprise later.

Wherever this project says "beta-neutral", it means neutral to the linear
market factor. It does not mean neutral to market convexity.

### Still open

- **The model itself.** `lambdarank` with the cross-section as query group is
  the design intent, but no hyperparameter has been chosen and none may be
  chosen on the holdouts.
