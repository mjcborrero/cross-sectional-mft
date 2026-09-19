# What this project found

A cross-sectional, beta-neutral crypto perpetuals strategy, built from scratch
with pre-registration and two sealed holdouts. Both holdouts are now spent.

**The strategy does not work.** The signal is real; the book does not monetise
it. This document records what was established, what was disproved, and what is
worth carrying forward.

---

## 1. The headline

| period | bars | Sharpe | ann return | maxDD | status |
|---|---|---|---|---|---|
| train (out-of-fold) | 2,755 | **+2.86** | +48.7% | −8.6% | in-sample-adjacent |
| holdout 1 (2025) | 1,095 | **+2.83** | +47.9% | −7.3% | spent |
| holdout 2 (2026) | 633 | **−0.35** | −2.8% | −12.4% | spent |
| 24h variant (2026) | 211 | **−1.34** | −15.9% | −17.1% | suggestive only |

Zero fees throughout, per the design assumption.

Two out-of-sample years disagree. The second is the cleaner test — it was run
after the hedge was fixed and after everything had been audited — and it is
negative.

---

## 2. The central finding: the signal survived, the book did not

This is the most useful thing the project established, and it is not obvious
from the Sharpe alone.

| | train | 2025 | 2026 |
|---|---|---|---|
| IC | +0.0489 | +0.0481 | **+0.0385** |
| hit rate | 53.6% | 54.3% | **55.0%** |
| **win/loss size ratio** | 1.12 | **1.063** | **0.805** |

The model ranked coins *better* in 2026 than in either prior period and still
lost money. Wins shrank 23%; losses did not. **The failure is in the
translation from ranking to position sizes, not in the ranking.**

Five alternative book constructions were tested on the identical signal — top-5
concentration, no inverse-vol, long-only, dispersion-scaled gross. **None
rescues 2026**; the best reaches +0.02. Concentration made it markedly worse
(top-5: −18.0%, maxDD −25.6%), which kills the natural hypothesis that the
signal lives in the tails and the middle is noise. It is the opposite.

---

## 3. What changed between 2025 and 2026

| | train | 2025 | 2026 |
|---|---|---|---|
| market return | +522% | −10.2% | **−31.9%** |
| market vol | 69% | 60% | **50%** |
| cross-sectional dispersion | 0.0175 | 0.0148 | **0.0123** |
| idiosyncratic vol | 0.0187 | 0.0147 | **0.0118** |

2026 was a **deeper, quieter bear**: 30% less dispersion than train. A
cross-sectional book eats dispersion, and the long-volatility payoff it
depended on evaporated:

| P&L by market-move quartile (bp/bar) | Q1 calm | Q2 | Q3 | Q4 wild |
|---|---|---|---|---|
| 2025 | +1.32 | +0.84 | +3.18 | **+9.60** |
| 2026 | +2.48 | −2.71 | +0.04 | **+1.67** |

May 2026 alone was −9.57% (worst bar −210bp). Excluding it, 2026 Sharpe is
+1.95 — recorded as a diagnostic, **not** as a result. "Excluding the worst
month it's fine" is the reasoning that killed the previous incarnation of this
project.

### Adding recent training data made it worse, not better

A last diagnostic: retrain through 2026-Q1 and re-run April onward, to test
whether the model simply needed fresher data to survive May.

| month | train ≤2025 | train ≤2026-Q1 | delta |
|---|---|---|---|
| 2026-04 | +2.09% | +1.42% | −0.67% |
| 2026-05 | −9.57% | −9.87% | −0.30% |
| 2026-06 | −1.82% | −2.22% | −0.40% |
| 2026-07 | +4.00% | +3.05% | −0.95% |
| **Apr–Jul** | **−5.74%** | **−7.90%** | **−2.16%** |

Every month got worse, not one. Sharpe over the window −2.16. Three extra
months of the most recent regime did not help the model handle May; it degraded
performance uniformly, which is what adding data drawn from a *different*
distribution does.

This was a diagnostic, not a test: the boundary was chosen after the bad month
was known, so a good result would not have been evidence. A bad one is still
informative, because "the model needed fresher data" is now ruled out.

---

## 4. Design findings that are true regardless of the P&L

These were measured, and they hold independently of whether the strategy makes
money.

**Dollar-neutral is not beta-neutral.** Equal long and short *dollars* leaves a
market exposure of `Σwᵢβᵢ`, which is zero only if weights are uncorrelated with
beta. Inverse-vol sizing overweights low-vol coins, which are also low-beta, so
the book ran a persistent unintended short: `corr(book, r_m)` = −0.127 at
t −6.71 on train. Per $100k gross that was **$3,500–$10,600 short the market**,
drifting, unchosen.

**Hedging with an overlay beats constraining the weights.** Forcing `Σwβ = 0`
moves the positions and loses the alpha with them (2.61 → 2.12). An overlay
leaves positions untouched and *improves* the book (2.61 → 2.86). These are
different operations and conflating them was an error.

**A rebalanced short does not earn the compounded path.** In 2025 the market
fell 10.4% but the *sum* of its 8h returns was **+7.2%** — 60% vol produces
~18 points of volatility drag. An 8h-rebalanced short therefore *lost* money in
a falling market. Static and rebalanced exposures are not the same instrument.

**The elegant hedge lost to the crude one.** `Σwᵢβᵢ` is knowable at trade time
with no lookback, which sounded strictly better than regressing on past
returns. It is not: that sum uses the *estimated trailing* β, so it cancels the
exposure the model *thinks* it has. In 2026 realised betas diverged and it sat
at t −3.97, while a 90-bar rolling regression on actual outcomes reached
t +0.01.

**Expanding windows cannot track a moving beta.** Over four years a new bar
moves an expanding estimate by ~1/4500. It held on train (t +2.92) and
collapsed out of sample (t −14.38).

| hedge | train t(r_m) | 2025 t(r_m) |
|---|---|---|
| expanding | +2.92 | **−14.38** |
| **rolling-90** | −1.19 | **+0.01** |
| ex-ante `Σwβ` | +0.67 | −3.97 |
| none | −7.23 | −14.89 |

**Market convexity is not hedgeable with perpetuals.** The book earns more when
the market moves hard *either* way (`corr(book, |r_m|)` +0.13, t +4.45 in
2025), and it is not opportunity-scaling: controlling for cross-sectional
dispersion, `|r_m|` survives at t +4.34 while dispersion is insignificant.
Vol-targeting was implemented and **rejected** — trailing vol explains 8.0% of
`|r_m|` variance on train and **0.2%** in 2025. You cannot size against what you
cannot forecast. Cancelling it needs a straddle.

**The 24h horizon is worse, not better.** IC +0.0138 vs +0.0385; Sharpe −1.34
vs −0.35; neutrality fails (t −3.90). Whatever edge exists is a *shorter*-horizon
effect. This disproves the hypothesis that the 8h clock traded faster than the
alpha decayed.

---

## 5. The feature programme

93 features across 9 mechanism families, every one gated and truncation-tested.

| stage | question | outcome |
|---|---|---|
| 0–1 | valid, point-in-time, ranks anything | all 93 pass |
| 2 | estimation quality | 1 flag |
| 3 | survives to disjoint data | **7 fail** |
| 4 | redundancy (within-bar rank) | 9 absorbed → 84 |
| 5 | implied turnover | priced, not a kill criterion |
| freeze | consolidated removal | **72 frozen** |
| 6 | univariate IC | 20 flagged |
| 7 | grouped permutation vs shadows | 6 of 67 groups stable |
| 8 | stability selection | **selection did not help** |

**Selection was a wash.** The pruned set beat all 72 in 2 of 4 folds and lost on
the mean (+0.0409 vs +0.0450). The regularised model already handles noise
columns. Only families **E** (regime) and **G** (spot-perp) survived
Bonferroni correction at the family level.

**Hyperparameter tuning bought nothing.** Apparent gain +0.0024; honest gain
under sequential selection **+0.0010**. Selection bias was +0.0014 — more than
half the apparent gain. The grid picked a different config nearly every fold,
which is the signature of ranking noise.

**The priors were a coin flip.** Among features whose IC is measurable (|t| ≥ 2),
**11 of 21 matched their pre-registered sign — p = 1.0000**. Ten significant
features were wrong-signed, including the strongest one in the screen. The
economic reasoning did not predict direction, so the surviving set had to be
treated as data-mined. Holdout 1 held up anyway; holdout 2 is the outcome that
expectation predicted.

**An asserted coverage claim was wrong 15 times out of 16.** Family H skipped
most of the inherited library as "already covered by designed families." Built
as Family I and tested, **15 of 16 survived clustering**. Assertions about
redundancy must be measured.

---

## 6. Bugs found, and the pattern in them

Ten defects were caught. **Two would have produced plausible, wrong results.**

| # | bug | how it surfaced |
|---|---|---|
| 1 | `funding_rate_change` measured 8h-boundary **timestamp jitter**, not funding — exactly 0 for every coin in 18.5% of bars | a coverage gate |
| 2 | Two Family I columns were `log()` of siblings — rank corr **1.0000000000** | identical persistence scores |
| 3 | Stage 5 turnover normalised by `k` not the realised book | an **impossible −12.8%** |
| 4 | `lambdarank`/NDCG was the wrong surrogate — IC **+0.0097** vs +0.0489 | model scored below a no-fit composite |
| 5 | `smooth()` bound defaults at **import**, so runtime config changes did nothing | two variants returned byte-identical results |
| 6 | Wall override moved `HOLDOUT1_START_MS` itself → **empty holdout** | crashed rather than reporting a number |
| 7 | `build_forward_returns` output name hardcoded → 24h build **overwrote the 8h artifact** | the success line named the wrong file |
| 8 | Cross-check hardcoded `market_index_8h` → compared different quantities | a 0.836 correlation that should have been 1.0 |
| 9 | `_OPEN_END` sentinel as an `arange` endpoint → **597 GiB** allocation | immediate crash |
| 10 | **Bar-count parameters** that change meaning with the clock | three separate instances |

### The recurring pattern

**Bug 10 appeared three times**, and it is the one to remember:

- `anatomy.py`'s `bar_duration_*` — dollar-bar durations, constant on a fixed clock
- `W30D = 90` — "30 days" only on an 8h grid
- `min_obs` in the weights builder — a *count*; 30 observations take 10 days at 8h and 30 days daily

Any parameter expressed in bars silently changes meaning when the clock changes.
On a daily grid the weights blanked for 29 days after each data gap, while the
8h build resumed after ~10 — computing a "30-day ADV" from 10 days of data. The
daily behaviour was the correct one.

### Two meta-lessons

**Impossible values are the cheapest detector.** Bug 3 was found because
turnover was negative, which cannot happen. The same mis-normalisation was
silently deflating every discrete-valued feature's cost while staying in range.

**A check that always fails stops being read.** The holdout seal was patched
from 2025 → 2026 and would have needed patching forever, because building a
feature panel over a holdout is a *prerequisite* to evaluating it. It was
rewritten around what actually spends a holdout: features are deterministic
functions of past data and spend nothing; letting *outcomes* inform a decision
does. Only the out-of-fold score panel is now gated.

---

## 7. Checks that earned their place

- **Truncation test** — build on full data and on truncated data; overlapping
  rows must be bit-identical. Caught nothing here because nothing was wrong,
  but it is the guarantee that makes everything else meaningful.
- **PIT-by-extension** — rebuild with a year of future data added; every prior
  row must be unchanged (max|diff| 0.00e+00 across all 9 families). Stronger
  than the truncation test: it asks whether the future changes the past.
- **Placebo** — random scores through the same book give Sharpe +0.005. Also
  yields the honest error bar: a *random* score on 2,755 bars ranges −1.15 to
  +1.29, **sd 0.559**. That is 3.5× the seed dispersion and the number that
  should have qualified every Sharpe quoted.
- **Independent recomputation** — rebuild each artifact by different code, not
  by re-running the builder. `fwd_ret` reproduced by compounding three 8h
  returns to 8.88e-16.
- **Input freshness** — every dataset must reach the panel's end. Caught stale
  funding (+696h) on its **first run**, which set the holdout-2 cap.
- **Sign check** — free, unfoolable, and it delivered the most important
  epistemic result in the project.

---

## 8. What is reusable

**Infrastructure, essentially all of it.** Purged walk-forward folds, the wall
mechanics with explicit spend tracking, the gate harness, both audit suites,
the data ingestion and manifest layer. This is regime-independent and was most
of the work.

**The finding that the cross-sectional signal has out-of-sample information.**
IC positive in both holdout years is not nothing. It is weak as a standalone
P&L engine and now has two years of evidence for that, but it is plausible as
an overlay — sizing a directional book, or choosing which coins to hold.

**The three context features this project discarded.** `btc_resid_lag`,
`eth_resid_lag`, `term_basis_slope` were exempted from every stage as
"bar-level, zero ranking power." That is exactly what makes them the natural
starting point for a *directional* model.

---

## 9. Honest limitations

- **No clean holdout remains.** Both are spent. Anything built on 2020–2026 now
  can only be validated by forward data.
- **The 24h test is suggestive, not out-of-sample.** 2026 was already seen, and
  a 24h return is three compounded 8h returns over the same prices.
- **Survivorship bias.** The 20-coin universe was chosen using present-day
  liquidity. The split does not address this and neither does anything else here.
- **Breadth caps any directional variant.** `IR ≈ IC × √breadth`. Cross-
  sectionally there are ~19 bets per bar; directionally there is 1. The same
  IC of 0.045 yields roughly **1/4.4 the Sharpe**. Matching this book's train
  result directionally would need IC ≈ 0.20, which is not plausible.
- **Sample size was always thin.** 72 features against ~4,560 independent bars
  is ~65 bars per feature. That constraint, not algorithm choice, ruled out
  greedy selection and bounded everything.

---

## 10. What the record shows about the process

The pre-registration worked. Every threshold that was later inconvenient had
been written down first, and the two occasions where a bar was moved
(`Gate 9A`'s slope test, Stage 3's change-tier) were recorded as post-hoc rather
than presented as design.

Nothing was tuned on a holdout. The one place selection touched spent data —
choosing rolling-90 over ex-ante on the 2025 column — is recorded in the spec
as such, and holdout 2 was the only clean test of it. **It passed**:
`corr(book, r_m)` = −0.024 (t −0.60) out of sample, the first time neutrality
held on unseen data.

The strategy failed. The method did what it was built to do, which was to find
that out before any money was involved.

---

## 11. Why the book lost while its IC stayed positive (2026-09)

Added after the ridge experiment. This is the clearest mechanical account of
the failure the project has, and it revises §3.

### Ridge was tried and lost

Same panel, folds, book, hedge and evaluation; only `fit`/`predict` changed.

| model | train IC | train Sharpe | 2026 IC | 2026 Sharpe |
|---|---|---|---|---|
| ridge α=1 | +0.0364 | +1.13 | +0.0312 | −1.38 |
| ridge α=1000 | +0.0396 | +1.14 | +0.0315 | −1.67 |
| ridge α=10⁵ | +0.0406 | +0.09 | +0.0389 | −1.40 |
| LightGBM | — | +2.86 | — | −0.35 |

Worse in both periods at every α, no plateau. **The learner was never the
problem.** Ridge also needs no seed ensemble (closed form) and neutral-fills
1.6% of ranks that LightGBM routes natively.

### The contradiction ridge exposed

Ridge's IC barely degrades out of sample (+0.036 → +0.031) yet its book loses
15%. It is not vol-normalisation: 2026 IC is positive in **every** space —
target +0.031, dollar residual +0.027, raw return +0.022. But the tails invert:
top-5 leg −1.57 bps, bottom-5 +3.75 bps.

Quintiles are non-monotone on **train** too: Q1 −0.34, Q2 −0.17, Q3 −0.39,
Q4 −0.02, Q5 +1.11 bps. The entire train edge sat in one bin, spread
+1.45 bps/bar.

### The mechanism

Splitting bars by cross-sectional dispersion of the forward residual:

| tercile | train Q5−Q1 | 2026 Q5−Q1 |
|---|---|---|
| quiet | **+9.11 bps** | **+9.27 bps** |
| mid | **+9.17** | **+10.27** |
| violent | −14.57 | **−35.52** |
| plain IC | +0.0412 | +0.0274 |
| dispersion-weighted IC | +0.0297 | **+0.0007** |

**The quiet/mid edge did not degrade at all** — ~+9 bps in both periods. The
whole failure is the violent tercile going −14.6 → −35.5.

Per-bar IC weights every bar equally; P&L weights by how much moved. The model
is right when little is at stake and wrong when a lot is. That is how a
positive IC coexists with a losing book — and it means **`IR ≈ IC·√breadth`
never applied to this book.** Any future go/no-go gate should use the
dispersion-weighted IC, not the plain one. This corrects the standard used in
`scripts/directional_ic.py`.

### Acting on it does not work

Dispersion *is* forecastable from strictly pre-`t_obs` data (rank corr 0.61
train / 0.38 in 2026; P(violent | called quiet) 11% / 19% vs 33% for a coin
flip). Gating the book on forecast dispersion, with expanding past-only
percentiles, shows a consistent dose-response across all three windows — more
gating, better 2026 — reaching +0.53 Sharpe at win3/keep-50%.

But the **paired** test on identical bars kills it. Across 12 cells the best
improvement is +0.85 bps/bar at **t = 0.98 (p = 0.33)**. Not one cell is
significant in the helpful direction; the only significant cell (win9 keep-90%,
t −1.98) is *harmful*, and is expected among 12 tests.

**So the diagnosis is solid and the treatment is not.** Knowing the edge lives
in low-dispersion bars explains the failure; it does not repair it, because the
forecast is too weak to separate the terciles cleanly enough at this sample
size. Gating converts −0.35 into roughly zero, not into a strategy.
