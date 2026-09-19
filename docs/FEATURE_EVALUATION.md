# Feature evaluation — the screening funnel

*Drafted 2026-08-07. Companion to `docs/FEATURE_EXPLORATION.md` (the idea
pool) and `docs/TARGET_DESIGN.md` (the label).*

**Status:** the target-free stages below are **pre-registered** — thresholds
fixed here, before any feature is built. The target-touching stages are
**NOT settled** and are recorded as open (§8).

---

## 1. The organizing principle

**Anything that does not look at the target is statistically free.**

Redundancy, persistence, coverage, leakage — none of these consume
multiple-testing budget, because none of them can be tuned toward a result
that does not yet exist. They can be run, adjusted, and re-run without
accounting.

The moment a test touches the target, the budget starts and the candidate list
must be frozen.

**So the funnel exists to arrive at first-target-contact with as few
candidates as possible.** Expected: ~80 constructions in, **~30 out**, zero
statistical budget spent.

Ordering within the target-free stages is by cost and kill rate — cheapest and
most decisive first.

---

## 2. Stage 0 — Validity and leakage

| check | threshold | kills |
|---|---|---|
| Coverage | ≥95% of rows non-null | NaN-heavy constructions |
| Global variation | std > 0 | constants |
| Finiteness | no inf / −inf | degenerate divisions |
| **Per-coin coverage** | **no coin below 80%** | hidden tilts |
| **Point-in-time assertion** | every input timestamp `< t_obs` | declared leaks |
| **Truncation test** | see below | *undeclared* leaks |

**Per-coin coverage matters more than it looks.** A feature 95% covered
overall but 0% covered for two coins is not the same as one evenly 95%
covered — the missing coins are systematically excluded from those
cross-sections, which is a survivorship-shaped tilt introduced by the feature
itself.

### The truncation test — the check I would least want skipped

Build the feature twice:

1. On the full train set → `f(t)`
2. On train **truncated** at some midpoint `T` → `f'(t)`

**For every `t ≤ T`, `f(t)` must equal `f'(t)` exactly.**

Deleting future data cannot change a genuinely point-in-time value. If it
does, the feature is using information from beyond `t`.

This catches what timestamp assertions cannot:

- full-sample normalisation (z-scoring by the whole sample's mean/std)
- any estimated parameter fitted on all data — PCA loadings, a global
  regression coefficient, a fitted threshold
- a rolling window that accidentally includes the current or future row

A feature can pass every timestamp check and still fail this one. Step 8's
Gate 8B is the same idea applied to the target, and it caught a 94% error.

**Where to place the cut — learned 2026-08-10.** The cut must fall *after every
coin has listed*. If it does not, the truncated build has a narrower
cross-section than the full one, and any cross-sectionally normalised feature
differs in the last bit: pandas sums a 20-column frame (one all-NaN) slightly
differently from a 19-column frame. Measured at **6.6e-16 median — one ULP** —
amplified to 1e-13 through a variance ratio.

That is not a leak. But the tempting response, loosening the gate to a
tolerance, would mean accepting 1e-13 differences *everywhere*, and a real
bounded lookahead is on the order of 1e-2. Removing the cause instead keeps
the gate at **exact equality**, which is the setting that actually catches
things. A later cut also compares more shared rows, so the test gets stronger
rather than weaker.

**Two limits of this test, worth knowing:** it cuts the whole dataset at one
date, so it catches full-sample contamination but **not a bounded per-row
lookahead** — a feature reading one hour past its own stamp survives it
untouched. That has to be prevented by construction (see `regime.py`, which
uses `px_obs` rather than `px_fill` for exactly this reason).

---

## 3. Stage 1 — Ranking power

Specific to the `lambdarank` objective, and free.

| check | threshold | kills |
|---|---|---|
| **Within-bar variation** | within-bar std > 0 in ≥95% of bars | bar-level constants |
| **Distinct values per bar** | ≥10 of ~19 | coarse categoricals |

**Within-bar variation is the check most often skipped.** A feature identical
for every coin in a cross-section has **exactly zero** ranking power — the
loss only sees within-bar ordering. Market-wide aggregates (dispersion,
average correlation, market funding) die here, for free, before costing
anything. They remain usable only as interaction terms
(`FEATURE_EXPLORATION.md` §12).

**Distinct values** is the finer version. A 4-state categorical ties roughly
five of nineteen coins together, and a ranking loss cannot order within a tie.
Continuous constructions preserve the full ordering; this threshold makes the
cost of a categorical visible before it is paid.

---

## 4. Stage 2 — Estimation quality

For any feature that is itself an *estimate* — a correlation, a beta, an AR
coefficient, a variance ratio.

| check | threshold | rationale |
|---|---|---|
| **SE vs cross-sectional spread** | SE < ½ the within-bar spread | a feature whose estimation noise rivals its cross-sectional signal is mostly measuring itself |
| **Construction sensitivity** | nudge the lookback ±15% (e.g. 30d→35d); within-bar rank correlation of old vs new **> 0.9** | if a small parameter change reorders the cross-section, the feature is fitting noise, not structure |

The SE check is what distinguished 8h/30d correlation (SE 0.105) from 8h/90d
(SE 0.061) in `FEATURE_EXPLORATION.md` §8.2 — same quantity, one usable and
one not.

Construction sensitivity is the cheaper cousin of a robustness study, and it
is target-free: a feature whose ranking survives a window nudge is measuring
something; one whose ranking flips is measuring the window.

---

## 5. Stage 3 — Persistence

**Non-overlapping cross-sectional rank persistence > 0.15.**

Does the ordering a feature produces survive to a period that shares no data
with the one it was measured on? Sample the feature at intervals at least as
long as its estimation window, then rank-correlate consecutive snapshots.

This is the stage that earned its place empirically. The correlation term
structure (`FEATURE_EXPLORATION.md` §8.2) looked like the best candidate in
its group on every other criterion — only +0.295 redundant with existing
columns, apparently almost pure new information — and scored **+0.038** here.
It was noise.

**Low redundancy has two explanations, new information and noise, and noise
correlates with nothing.** No redundancy check can separate them. Only
persistence can.

Reference points from measurements already taken: `corr₈ₕ(90d)` scored +0.411
(real), the term structure +0.038 (noise).

---

## 6. Stage 4 — Redundancy, at three levels

Pairwise correlation alone is not enough, because a feature can be a linear
combination of three accepted features while correlating below threshold with
each individually.

| level | method | threshold |
|---|---|---|
| Pairwise | \|corr\| against each accepted feature | < 0.90 |
| **Multivariate** | R² of the candidate regressed on **all** accepted features | < 0.80 |
| **Clustering** | correlation-cluster the entire candidate set, keep one representative per cluster | — |

**Clustering is where the volume reduction actually happens.** The idea pool
contains many variants of one idea — four lookbacks of residual momentum are
*one cluster*, not four candidates. Clustering handles that systematically
rather than by judgment, and judgment is exactly what a frozen list is meant
to remove.

Choose the representative by *interpretability and cost*, not by any
target-related quantity — selecting the cluster member with the best IC would
be target contact smuggled into a target-free stage.

§8.2 of `FEATURE_EXPLORATION.md` records what this stage is for: rolling
correlation to BTC turned out to be **95% redundant** with a column Step 6
already emits.

---

## 7. Stage 5 — Implied cost

**Not a kill criterion.** Priced information, obtained free.

Rank autocorrelation at lag 1 → implied turnover:

- **near 1.0** — a static tilt. Ranks the same coins the same way every bar;
  cheap to trade, but carries no timing
- **near 0** — churn, paid for in slippage even under the zero-fee assumption

Record it for every survivor. It becomes decisive later when two features have
similar value and different cost.

### MEASURED 2026-08-24 — 84 survivors priced

`scripts/stage5_turnover.py`. Rank autocorrelation between consecutive 8h
instants, plus the turnover it implies for a top-4/bottom-4 book rebalanced
every bar. Three context columns exempt: no ordering, so no book.

| band | n | reads as |
|---|---|---|
| ≥ 0.95 | 25 | static tilt — 1–5% turnover, nearly free, no timing |
| 0.80–0.95 | 9 | slow — 10–25% |
| 0.50–0.80 | 17 | medium — 30–50% |
| < 0.50 | 30 | fast — 60–95%, the expensive end |

**The cheapest and the most expensive are both worth suspicion.** The
sector/correlation block prices at ~1% turnover because it barely reorders
anything — cheap precisely because they carry no
timing. At the other end `basis_acceleration` (−0.593, **95.1%**) and
`perp_spot_return_gap` (−0.424, 90.5%) reverse their own ordering every bar:
negative autocorrelation is a second difference behaving like one, and the book
is rebuilt from scratch each rebalance.

The four `resid_lag_*` / `resid_reversal_8h` features sit at −0.023 and 77.2%,
which is the correct signature for a transient state and not a defect — the
same reasoning that exempts them from Stage 3.

> **Turnover must be normalised by the REALISED book, not by k.** A tie at the
> cutoff puts every tied name in the book, so `rank <= k` can select more than
> k. The sector tier takes one value per sector — six per bar — so
> `sector_cohesion` selects a median of **5** names for a "top-4" book, and
> dividing by k returned **−12.8%**. Negative turnover is impossible, and it
> was the only reason the error surfaced: the same mis-normalisation was
> quietly deflating every discrete-valued feature's cost without ever leaving
> the valid range (`funding_sign_persist` 13.4% → 10.3%, `roll_spread`
> 52.3% → 44.4%, `funding_rate_lag_1/2` 54% → 43%). Now divided by
> max(|Sₜ|, |Sₜ₋₁|), and the realised book size is printed so ties stay visible.

---

## 8. The freeze point, and what is NOT yet settled

**Between Stage 5 and any target contact, the candidate list is frozen** —
names, constructions, parameters, and expected signs, written down. Stages 0–5
may be re-run freely up to that moment. Nothing after it may be re-run without
being counted.

### Target-touching stages — OPEN

Deliberately unsettled. What is agreed:

- **The univariate screen must be lenient, not selective.** Its job is to
  remove features that are provably nothing, not to pick winners. Selecting
  winners univariately overfits, and it discards features that only work in
  combination — `crowd_winning` (`FEATURE_EXPLORATION.md` §1.1) may be flat
  univariately and sharp conditionally.
- **Multiple-testing correction counts mechanism families, not columns.** Six
  families is a Bonferroni factor of 6 (z ≈ 2.64). Sixty columns is a factor
  of 60 (z ≈ 3.29), which would reject everything, since the benchmark signal
  itself is only 3.6σ.
- **The noise floor is already established**: IC SE **0.00366**, so \|IC\| >
  0.0072 for 2σ. Reversal alone scores **0.0133**. Zero is not the bar.

What is **not** agreed:

- The regime-dependence screen. A first attempt was rejected as underpowered:
  splitting 4,558 bars in half gives a gap SE of 0.00732, so the minimum
  detectable gap (0.0146) **exceeds the entire benchmark signal** (0.0133). It
  can only catch features that completely flip between regimes. That may still
  be worth running — gross failure is what kills strategies — but it must be
  reported as a gap with a confidence interval, never as a pass/fail, and
  never described as showing regime independence.
- The final selection method.
