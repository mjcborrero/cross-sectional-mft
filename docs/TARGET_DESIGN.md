# Target design — cross-sectional, beta-neutral

*Drafted 2026-08-05. Builds on `docs/BETA_NEUTRAL_DESIGN.md`, which fixes the
market factor and the neutrality requirements. Nothing here is implemented.*

*Revised 2026-08-05: holding period changed from 12h to **8h** (funding
alignment; 12h dropped, 24h retained as a documented alternative); added an
explicit execution-lag parameter and the three-timestamp model; added a
dedicated section on the decision clock.*

This document specifies the prediction target and lays out a **gated,
step-by-step build plan**. Each step has a pre-registered pass criterion. **No
step begins until the previous step's gate has passed.** A failed gate is a
finding to investigate, not an obstacle to route around.

---

## 1. The target

Three timestamps, not one — see §3 for why this distinction is load-bearing:

| Symbol | Meaning |
|---|---|
| `t_obs` | Decision time — newest data the model is allowed to use |
| `lag` | Modeled delay between deciding and actually holding the position |
| `t_fill` | `t_obs + lag` — when the position is treated as established |

```
t_i,t_obs  =  ( r_i,[t_fill, t_fill+h]  −  β̂_i,t_obs · r_m,[t_fill, t_fill+h] )  /  σ̂_ε,i,t_obs
```

| Symbol | Definition |
|---|---|
| `r_i,[t_fill,t_fill+h]` | Forward return of coin `i`, measured from the actual fill instant |
| `r_m,[t_fill,t_fill+h]` | Forward return of the volume-weighted universe index (D1), same window |
| `β̂_i,t_obs` | Point-in-time, shrunk beta of coin `i` to `r_m`, estimated on data strictly before `t_obs` |
| `σ̂_ε,i,t_obs` | Point-in-time **idiosyncratic** volatility of coin `i`, same estimation cutoff |
| `h` | Holding period. **8 hours (primary)**; 24h retained as a documented alternative (§4) |
| `lag` | Execution delay. **Default 60 minutes** (§3) |

Every estimated quantity (`β̂`, `σ̂_ε`) uses data strictly **before `t_obs`** —
not before `t_fill`. This is intentionally conservative: `t_obs` is the actual
moment the decision is made, and using anything after it is lookahead
regardless of when the position ends up filling. The forward-return legs, by
contrast, are measured from `t_fill`, because that is when the position is
genuinely live.

The target is ranked **within each cross-section** (all coins observed at the
same `t_obs`), with the cross-section as the query group.

Interpretation: `t` is a standardized residual return — an idiosyncratic
information ratio, adjusted for how long it actually takes to get into the
position.

---

## 2. Holding period — **DECIDED**

**`h` = 8 hours, primary. Rebalance every 8h, hold 8h, label horizon 8h.**
**24h is retained as a documented alternative** for later comparison; 12h has
been **dropped**.

### Why 12h was dropped

Binance USDT-M funding settles every **8 hours** — 00:00, 08:00, 16:00 UTC,
three events per day. A 12h period cannot contain them evenly:

- `00:00 → 12:00` contains **one** funding event (08:00)
- `12:00 → 00:00` contains **two** (16:00, 00:00)

Funding is excluded from the *target* but charged in P&L (§5). With 12h
periods, the two daily rebalances would carry systematically different
funding costs, alternating for reasons that have nothing to do with signal —
a spurious morning-vs-evening difference baked into the backtest. This is
exactly the kind of contamination the method discipline in §9 exists to
catch, so it is better avoided than discovered.

### Why 8h over 24h

Both horizons align cleanly with funding (8h = exactly one event, 24h =
exactly three). The remaining trade-off is sample size. Research runs on
**train only** (2020-01-01 → 2024-12-27, ≈5.0 years); the effective sample is
the number of **non-overlapping cross-sections**, not the row count:

| Horizon | Independent cross-sections | Features supportable at ~40:1 | Funding events/period |
|---|---|---|---|
| 4h | ~10,900 | ~270 | 0 or 1 (uneven) |
| **8h** | **~5,470** | **~135** | **exactly 1** |
| 12h | ~3,640 | ~90 | 1 or 2 (uneven) — dropped |
| 24h | ~1,820 | ~45 | exactly 3 |

8h gives 3× the independent sample of 24h and a materially larger feature
budget, at the cost of a third rebalance per day. Under the zero-fee
assumption that cost is turnover and slippage, not fees.

24h is kept as a documented alternative rather than discarded, because carry
and positioning mechanisms plausibly decay more slowly and may express
differently over a full day. If 8h underperforms in a way that looks like
"too little signal survives the horizon," 24h is the first thing to
re-examine — not a new feature family.

---

## 3. Execution lag — **DECIDED to model explicitly**

### The three-timestamp problem

Treating "decision time" as a single instant hides three different things:

| | |
|---|---|
| `t_obs` | Data cutoff — newest information the model may use |
| `t_dec` | When the signal is actually computed |
| `t_fill` | When the position is actually established |

These are never equal in a live system, and collapsing them in the target
silently claims an edge that cannot be captured.

**Kline completeness.** The candle covering `[t, t+1min)` does not exist until
just after `t+1min`. Using "the `t_obs` kline" as if it were available at
`t_obs` uses a candle that closes after the decision point — and that closing
minute is the first minute of the forward return. A small, free edge that
does not exist.

**Funding-instant ambiguity.** Funding settles at exactly `00:00:00`,
`08:00:00`, `16:00:00`. Deciding to rebalance at that exact instant makes
whether you pay it depend on microsecond fill timing — unknowable, and
unmodelable in a backtest. Any convention adopted here ("old book holds
through settlement, new book starts after") *is* an execution lag, just an
undocumented one.

**Funding-instant microstructure.** Other participants position around
funding timestamps to collect or avoid payments. Filling into that flow is
close to the worst moment in the window to trade.

### The realistic size of the gap

The old live pipeline's own cycle ledger gives a measured lower bound:

```
backfill      ~10 min
live-update   ~15–28 min
bars          ~0.5 min
live-panel    ~24–33 min
targets       ~2–3 min
──────────────────────────
              ~50–65 min, data cutoff → execution start
```

A target that assumes `lag = 0` (observe and fill at the same instant) claims
an edge that this pipeline has never been able to capture. `lag` is therefore
an explicit, first-class parameter of the target — not an implementation
detail decided later.

**Default: `lag = 60 minutes`.** Conservative relative to the measured
50–65 minutes; deliberately not the aggressive end.

### Revision (2026-08-31) — the 60 min was sized for the wrong architecture

The ledger above measures a **sequential batch pipeline**: backfill, then
live-update, then bars, then live-panel, then targets. That is an artifact of
how the old system was built, not a property of the problem. Data collection
and feature computation are independent and can run concurrently and
incrementally; a streaming ingest with incremental features puts `t_dec`
within seconds of `t_obs`. Roughly 95% of the 50–65 minute budget is
pipeline serialisation, and it does not survive a rewrite.

Three components of the lag are **not** pipeline latency and do not
parallelise away:

| | size | why it is irreducible |
|---|---|---|
| Kline completeness | ~1 min | `[t, t+1min)` does not exist until after `t+1min` |
| Fill-price coherence | ~2 min | `build_forward_returns.asof()` takes the last close **strictly before** `t_fill`. Below 2 min the entry price is dated at or before the decision instant that caused the trade — not conservative, incoherent |
| Funding-instant microstructure | judgement | `t_obs` sits exactly on 00:00/08:00/16:00 settlement, the worst instant in the window to cross |

**`lag` is therefore a first-class CLI parameter with a hard floor of 2 minutes
and a defensible default of 5.** `run_strategy.py --lag` refuses anything below
the floor.

**The lag is threaded through the price grid, not just the label.** Rebuilding
at a new lag requires the whole stack — decision grid → weights → index →
betas → forward → target — because `px_fill` feeds the index, which feeds the
betas. The *features* are unaffected: every featdef reads `px_obs` and never
`px_fill` (see `mft/featdefs/regime.py`), so the frozen list stays valid.
`scripts/check_lag_rebuild.py` asserts both halves — that `px_obs` and the
weights are bit-identical, and that `px_fill`, `r_m`, `fwd_ret` and `target`
genuinely moved.

**What a shorter lag will do to the measured numbers, stated in advance.**
Step 7L already measured it: the probe reversal IC runs 0.0133 at lag 60 and
0.0243 at lag 5, and only 54.6% of it survived to lag 60. Cutting the lag
roughly **doubles** measured IC. That is *not* new alpha — it is the
short-horizon reversal component the 60-minute lag was excluding, returning.
It also compounds with this project's zero-fee assumption in the same
direction, and reversal at short latency is precisely the family that dies of
transaction costs. Any improvement seen here is a recovered assumption, not a
discovery, and **both holdouts are already spent**, so re-running 2025/2026 at
a new lag is a diagnostic and not an out-of-sample test.

### Measured outcome (2026-08-31) — the lag was not the constraint

Full stack rebuilt at `lag = 5` and run against the identical frozen spec,
72 frozen features, seed ensemble and `--end 2026-07-31` cap:

| | lag 60 | lag 5 |
|---|---|---|
| train Sharpe | +2.86 | **+3.07** |
| 2026 Sharpe | −0.35 | **−0.80** |
| 2026 ann return | −4.17% | −8.76% |
| 2026 hit rate | 54.98% | 52.92% |
| 2026 turnover | 86.2% | 90.3% |

**Better in-sample, worse out-of-sample.** That is the textbook overfitting
signature, and it settles the question: the 60-minute lag was not what was
holding this strategy back. The extra edge recovered at short latency is
train-specific and does not generalise — consistent with it being the
short-horizon reversal component, which is exactly the part most likely to be
sample-specific and least likely to survive costs.

Note also that the doubling predicted from Step 7L's probe did **not**
materialise on the book: +0.20 Sharpe on train, not +100%. The probe measured
a reversal signal; the 72 frozen features are not primarily reversal signals,
so the lag was suppressing something this book was barely using.

**Consequence for the live design.** A faster pipeline is still worth building
on its own merits — it widens the set of strategies reachable later — but it
is not a fix for this one, and `lag = 60` remains the recorded default because
lowering it did not improve genuine out-of-sample performance.

### Gate 3L — lag sensitivity (new; sequenced into the build plan as Step 7L)

Build the target at `lag ∈ {5, 15, 30, 60, 120}` minutes, everything else
held fixed, and measure how the target's predictive relationship with
realized forward returns decays with `lag`.

**Why this matters beyond picking a number.** If the edge is gone by
`lag = 60`, the strategy has no viable execution path at this pipeline's
current speed, and that needs to be known **before** months of feature work,
not after. If the edge survives comfortably to `lag = 120`, that is
operational slack worth having on the record.

*Expected:* some decay with increasing lag; the question is whether it is
gentle (feature/signal timescale genuinely exceeds execution delay) or sharp
(the edge is a microstructure effect that does not survive realistic
latency). No result is assumed in advance here — this is closer to a
measurement than a pre-registered pass/fail gate, and its outcome is itself
information about what kind of edge, if any, is present.

---

## 4. The decision clock

Three distinct clocks are in play in a system like this, and conflating them
is what made the old pipeline complicated. They are independent choices:

| Clock | Governs | Status here |
|---|---|---|
| **Decision clock** | When rebalancing happens; when labels start/end | **Forced**, see below |
| **Estimation clock** | Return frequency used for `β̂`, `σ̂_ε` | Open — resolved by Gate 4 (Epps check) |
| **Feature clock** | How features are sampled | Out of scope here; can be anything, then resampled onto the decision clock |

### The decision clock is wall-clock, and that is forced, not chosen

Once `h` is a funding-aligned wall-clock duration, there is no freedom left.
The label is a wall-clock forward return, the holding period is wall-clock,
and funding settles on wall-clock. Every economically meaningful quantity
here is already wall-clock, so a volume clock for the *decision* grid would
be measuring activity time against a target defined in calendar time —
internally inconsistent.

**Decision grid: `t_obs ∈ {00:00, 08:00, 16:00} UTC`, each with a small
buffer for kline completeness (e.g. +1 minute) before the cutoff is treated
as final.** `t_fill = t_obs + lag` (§3). Windows remain exactly 8h apart and
each still contains exactly one funding event.

### Why the old volume-clock (dollar bar) approach is dropped, not kept as an option

Volume/dollar clocks exist because activity-time sampling produces returns
closer to IID with more stable variance than calendar time — a real
advantage in general. It does not survive contact with a **cross-sectional**
strategy specifically:

**Dollar bars are per-coin, so they close at different times for each coin.**
That is fatal for ranking: if coin A's bar closes at `10:05` and coin B's at
`10:15`, "the same bar" contains ten extra minutes of information for B that
A does not have. Ranking on that manufactures spurious cross-sectional signal
that has nothing to do with either coin's actual behavior.

The old system's workaround — using BTC's dollar bars as a **global** clock
and resampling every other coin onto it (hence the `bar_id` global key,
144,511 entries) — solves the synchronization problem but at a cost worth
naming: what results is an *irregular wall-clock* driven by BTC's own volume
profile, and no individual coin (BTC included, once resampling is involved)
actually gets to trade on its own activity clock. The complexity of bar
construction, pseudo-bar resampling, and the global `bar_id` mapping is paid
in full, and the core benefit — variance stabilization via each asset's own
activity time — is not collected.

**Most of that benefit is already captured elsewhere in this design.**
Dividing the target by `σ̂_ε` is itself a form of activity normalization: a
quiet 8h window and a violent 8h window are scaled differently before
ranking. That is the same subordination logic a volume clock is reaching
for, applied at the point where it actually matters for a cross-sectional
ranking — after the window closes, not before.

### What this removes from the build

For the target pipeline specifically, **no bar construction is needed at
all**:

- ~~Dollar bars~~ — take the kline close at each grid timestamp
- ~~Pseudo-bars~~ — the grid is synchronized across coins by construction
- ~~Global `bar_id` mapping~~ — the wall-clock timestamp *is* the key
- ~~BTC as everyone's clock~~ — gone; every coin is on the same explicit grid

`klines → 8h grid → returns → index → β̂, σ̂_ε → target`. Bar construction was
a feature-engineering choice inherited from the old pipeline, and features
are not needed until Step 10 (§8).

### Two hazards that come with a fixed grid, and their policy

**Missing data must never be forward-filled.** If a coin has a gap or outage
at a grid timestamp, forward-filling its price manufactures a zero return
while every other coin moves — the coin ranks mid-pack on fabricated data,
and the *next* observed return then contains the whole accumulated jump,
which looks exactly like signal. **Policy: exclude the coin from that
cross-section entirely, and log the exclusion.** Never impute a decision-grid
price.

**Minimum cross-section size must be enforced, not assumed.** After
exclusions, some grid timestamps will have fewer than 20 coins. A ranking
over 6 names is not comparable to one over 20 and should not be pooled with
it silently. **Policy: set a floor (candidate: 15), assert it, and drop any
cross-section below the floor** rather than ranking a thin one. This is
folded into Step 0's gate below.

---

## 5. Carried forward from before

Standing decisions, retained:

- **Ranking objective** (`lambdarank`), cross-section as query group. The book
  is top-k / bottom-k, so the loss should optimize *order*, not squared error
  — regression spends capacity fitting the middle of the distribution, which
  is never traded.
- **Funding excluded from the target, but charged in P&L.** The target
  predicts price only, so the ranking is not dominated by carry (which is
  highly persistent and would let the model score well by ranking on current
  funding). Funding is still paid when holding, so backtest P&L must include
  it as a cost. Target and P&L accounting are deliberately different things
  here.
- **No triple-barrier labels.** A variable, path-dependent holding period
  breaks the label/construction horizon match and makes the cross-section
  ragged — coins labeled over different windows are not comparable in a
  ranking.

---

## 6. What is forced, and what was chosen

Most of this target is **derived** from the beta-neutral base and from
operational reality, rather than picked freely:

| Element | Forced by | Free choice? |
|---|---|---|
| Numerator is a beta residual | Book is beta-neutral; target must neutralize what the book neutralizes | No |
| `r_m` = volume-weighted index | D1 of the base design | No |
| Label horizon = holding period | Target/construction consistency | No |
| Denominator is **idiosyncratic** vol | Sizing is inverse-idiosyncratic-vol | No |
| Decision clock is wall-clock | `h` is funding-aligned; funding settles on wall-clock | No |
| `h` ∈ {8h, 24h} | Funding alignment (§2) | Restricted, not fully free |
| `h` = 8h specifically (primary) | Sample-size argument, given the funding-aligned set | **Yes** |
| Execution lag is modeled | Measured pipeline latency; ignoring it claims an uncapturable edge | No (that it's modeled); yes (the default value) |
| Ranking objective | Book is top-k/bottom-k | Mostly forced |

### Why idiosyncratic vol, not total vol

After residualizing, what remains is `ε_i`, whose natural scale is `σ_ε,i`.
Dividing a residual by *total* volatility systematically under-normalizes
high-beta coins — their total vol is inflated by the very systematic
component just removed from the numerator. Scaling by a yardstick that still
contains the removed quantity is internally inconsistent.

This also closes the loop with sizing: in a beta-neutral book, the risk a
position contributes is its *idiosyncratic* risk. **Sizing is
inverse-idiosyncratic-vol** (DECIDED), so target, scaling, and sizing all
reference one consistent measure of risk.

---

## 7. Known hazard — beta estimation error leaks market direction

Recorded prominently because it is the most likely way this target fails
silently.

`β̂_i,t_obs` is necessarily ex-ante (realized beta over the holding window
would be lookahead). The true relationship over `[t_fill, t_fill+h]` uses the
*realized* beta, so the "residual" actually contains:

```
ε_true  +  (β_true − β̂) · r_m
```

The error term varies per coin and **scales with market direction**. In an
up-market, coins whose beta was underestimated appear to have positive
alpha. This is beta-in-disguise re-entering through the *estimator* rather
than through any feature — and no amount of feature screening will catch it.

Two consequences, both binding:

1. **Shrinkage is mandatory**, not optional — it directly reduces `β̂`
   variance.
2. **The regime split must be run on the target itself**, not only on
   features. If up-market and down-market IC differ materially on a
   supposedly residualized target, the estimator is the problem. This is
   Gate 9B.

---

## 8. Build plan — sequential, gated

Format for each step: **objective → procedure → gate (pre-registered, with
expected result) → on failure**. All work on the **train split only**.

The purge boundary is now `h + lag` (previously stated as `h` alone) — a
label window starting at `t_fill = t_obs + lag` and running `h` further must
still resolve before the holdout wall. At `h = 8h`, `lag = 60min`, purge =
**9 hours**.

---

### Step 0 — Price and return foundation, on the decision grid

**Objective.** Build klines resampled onto the `{00:00, 08:00, 16:00} UTC`
grid and per-coin returns. All derived data was purged; this is the
prerequisite for everything below. No bar construction (§4).

**Gate 0.**
- 20 symbols present; coverage 2020-01-01 → 2024-12-27
- No duplicate `(symbol, t_obs)`; no gaps beyond documented listing dates
- All returns finite; no zero-variance coin over any estimation window
- Missing-data policy enforced: no forward-filled decision-grid price (§4)
- Minimum cross-section size enforced: every retained `t_obs` has ≥15 coins
  (§4); count and report how many cross-sections are dropped below the floor
- `splits.assert_train_only()` passes, with the split's purge set to `h + lag`

*Expected:* coverage matches per-symbol listing dates (SUIUSDT from 2023-05,
all others by 2021-03); very few cross-sections dropped by the floor given a
20-coin universe with no history of majors delisting.

**On failure.** Fix the data layer. Nothing downstream is meaningful.

#### Result — PASSED 2026-08-05

`scripts/build_decision_grid.py` → `DATA_DIR/grid/decision_grid_8h.parquet`
**90,023 rows · 20 symbols · 4,680 cross-sections · 2020-09-18 → 2024-12-31.**
Independently audited by `scripts/audit_decision_grid.py`, which re-derives
prices from the raw monthly files through a separate code path (direct parquet
reads and pandas masks, not the project loader or `searchsorted`) — because a
pipeline agreeing with itself proves nothing.

**Known data limitation, permanent.** Eight symbols (XRP, SOL, TRX, XLM, LTC,
HBAR, NEAR, ZEC) are missing from the Binance archives for two windows:

```
2022-02-26 → 2022-02-28   (their 2022-02 dumps end at Feb 25)
2022-04-01 → 2022-04-02   (their 2022-04 dumps start at Apr 03)
```

This is upstream truncation, not an ingestion failure — BTC and ETH have
complete files for the same months. Effect: 17 cross-sections fell below the
15-symbol floor and were dropped whole (0.36% of the sample; each affected
coin loses 0.31–0.41% of its own span). **Any β or σ_ε estimation window
spanning these dates has ~5 fewer days of data for those eight coins**, which
matters for Step 5 window selection and should not be rediscovered later as a
mystery.

**Two audit findings that turned out to be defects in the audit, not the data**
— recorded because the reasoning is the useful part:

1. *Duplicate `close_time` across a symbol's files.* Real, but entirely after
   2026-06-01 (monthly dumps overlap the daily/live files once a month
   closes), with byte-identical values, and the builder filters to
   `close_time < wall` first. The check was auditing the whole tree rather
   than the region under test.
2. *254 exactly-zero 8h returns.* Not staleness. Prices are discrete, so a
   genuine round-trip can land back on the same tick — which is why low-priced
   coins show many (ADA 28, XRP 23) and BTC shows exactly one. Verified
   against raw klines: **all 254 intervals had real intra-interval movement**
   (median 1.59% high-low range, minimum 0.55%), zero frozen series. The check
   now tests for frozen series, which is the actual failure mode a stale or
   forward-filled quote would produce.

Price levels reconcile against known history (BTC $107,360 peak / $94,832 at
2024 year-end, ETH $4,852 ATH, SOL $1.20 launch floor, DOGE $0.70 May-2021
peak), ruling out unit errors that structural checks cannot see.

---

### Step 1 — Point-in-time universe and volume weights

**Objective.** Weights for the market index, using only trailing information.

**Gate 1.**
- Weights sum to 1 each period
- No coin carries weight before its listing date
- Weights at `t_obs` are computed from data strictly before `t_obs`

*Expected:* BTC weight dominant early (0.4–0.7), declining as alts grow.

**On failure.** A weight series built with present-day knowledge is
lookahead and silently contaminates every downstream quantity.

#### Result — PASSED 2026-08-05

`scripts/build_market_weights.py` → `grid/market_weights_30d.parquet`.
**89,443 rows · 4,651 cross-sections** (29 lost to 30d warm-up). Trailing 30d
*mean* of 8h quote volume — a mean rather than a sum, so the 2022 archive gap
degrades a weight gracefully instead of halving it for a month. Verified
through a separate path: 300 `adv` values recomputed from the grid, 0
mismatches.

**Finding that bears on D1.** The index is far more concentrated than the
20-coin universe suggests:

| | mean weight |
|---|---|
| BTCUSDT | **0.508** |
| ETHUSDT | **0.242** |
| next largest (SOL) | 0.047 |

**Effective N = 3.1 of 20 coins**, stable every year (BTC 0.49–0.60). D1 chose
a volume-weighted index over BTC alone because BTC misses the alt-specific
common factor — but weighted by Binance perp volume, the index is 75% BTC and
ETH. Whether that still buys anything over plain BTC is exactly what Gate 2
was pre-registered to decide.

---

### Step 2 — Market index `r_m`

**Objective.** Construct the volume-weighted index return series, on the
decision grid.

**Gate 2.**
- **Weights are lagged**: `r_m` at grid step `t` uses weights fixed at the
  previous grid step, not contemporaneous weights. Using contemporaneous
  weights is a classic lookahead and must be asserted against.
- `corr(r_m, r_BTC)` high but strictly below 1

*Expected:* `corr(r_m, r_BTC)` ≈ 0.85–0.95. A value at ~0.99 means the index
is effectively BTC and D1 gains nothing; below ~0.75 warrants investigation.

**On failure.** Revisit weighting before proceeding.

#### Result — PASSED 2026-08-05

`scripts/build_market_index.py` → `grid/market_index_8h.parquet`.
**4,648 steps · 2020-09-28 → 2024-12-31 · 69.1% annualised vol.**

**corr(r_m, r_BTC) = 0.9472** — inside the pre-registered band [0.75, 0.99).
**D1 survives.** Despite BTC carrying 51% of the weight, the index is not
merely BTC: it is materially differentiated, and the pre-registered "index IS
BTC" tripwire at 0.99 did not fire. For scale, an equal-weighted index would
give corr 0.8275, so volume weighting costs some differentiation but does not
collapse it.

| year | corr BTC | r_m vol | BTC vol |
|---|---|---|---|
| 2020 | 0.9158 | 70.2% | 60.9% |
| 2021 | 0.9407 | 93.9% | 83.3% |
| 2022 | 0.9636 | 70.6% | 60.6% |
| 2023 | 0.9569 | 46.4% | 44.1% |
| 2024 | 0.9494 | 55.6% | 50.5% |

Two correctness properties, both asserted rather than assumed:

- **Weights are lagged exactly one step.** `r_m` at step `t` uses weights
  stamped at `t-1`, which were built from volume observed before
  `t_obs(t-1)` — before the return window even opens. Contemporaneous
  weights would let the index know which coins traded heavily during the
  period it is measuring.
- **Gaps are not returns.** The 2022 holes leave consecutive *rows* 3.7 and
  2.3 days apart. Differencing rows blindly would book a multi-day move as
  one 8h return — a fake outlier landing in every beta window spanning it.
  38 (symbol, step) pairs were detected and excluded.

Verified independently: `r_m` recomputed from scratch for 200 steps, 0
mismatches; and compounding is exact on gapless years — 2023 and 2024 compound
to their point-to-point returns to the basis point (+156.87%, +122.46%),
proving no missing or duplicated steps.

**Caveat on 2022 cumulative figures.** 2022 has 1,076 of 1,095 steps, so
compounded 2022 returns omit the gap windows and understate the year:
BTC compounds to −68.75% against a true −65.04%, a **−3.71pp** shortfall
(the excluded windows contained BTC's Feb 26–28 rally). Point estimates of
*volatility* and *correlation* are unaffected; only cumulative 2022 return
figures are. Do not quote them as performance.

---

### Step 3 — Factor structure test  *(gates single vs two-factor)*

**Objective.** Determine whether one factor is sufficient. Pre-registered in
`BETA_NEUTRAL_DESIGN.md` §4; restated here as the gate.

**Procedure.** Correlation matrix of coin returns on train; eigenvalues;
report variance share of PC1/PC2/PC3 and `corr(PC1, r_m)`.

**Gate 3.**
- **PC2 ≥ 15%** → adopt two-factor neutralization; `β` becomes a vector and
  the target numerator gains a second term
- **PC2 < 15%** → single factor confirmed; proceed as specified
- **`corr(PC1, r_m)` < 0.90** → D1 is not capturing the dominant factor;
  return to the base design

*Expected (recorded in advance):* PC1 ≈ 70–85%, PC2 ≈ 5–12%,
`corr(PC1, r_m)` > 0.95.

**On failure.** This gate changes the target's functional form. It must be
resolved, not deferred.

#### Result — PASSED 2026-08-05: **SINGLE FACTOR**

`scripts/factor_structure.py` → `grid/factor_structure.json`. Two samples
reported together, never the better one:

| sample | coins | steps | PC1 | PC2 (95% CI) | corr(PC1, r_m) |
|---|---|---|---|---|---|
| ALL20 | 20 | 1,823 | 55.4% | **5.4%** [4.8, 6.3] | +0.9332 |
| LONG | 18 | 4,593 | 62.8% | **4.6%** [4.2, 5.1] | +0.9505 |

**Verdict: single-factor neutralisation. `β` stays a scalar; the target keeps
its specified form.** PC2 is 4.6–5.4% against a 15% threshold, and the *upper*
confidence bound (6.3%) is still far below it — so the call is unambiguous, not
a point estimate straddling the line. Both samples agree.
`corr(PC1, r_m) ≥ 0.933` confirms D1's index captures the dominant factor.

**Deviation from the pre-registered prior, recorded.** PC1 came in at
**55–63%**, below the expected 70–85%. Investigated rather than accepted:
compounding to longer horizons does *not* raise it (8h 62.8% → 24h 59.8% →
72h 56.0% → 7d 55.2%), and mean pairwise |corr| *falls* with horizon
(0.600 → 0.517) — the opposite of an Epps-style artifact. The prior was simply
wrong for this universe. **Crypto is less single-factor dominated than assumed,
which is favourable here: more idiosyncratic variance is more for a
cross-sectional strategy to trade.**

*Bearing on Step 4:* these 20 perps are the most liquid on Binance and trade
continuously, so non-synchronous trading appears not to bite at 8h and coarser.
Step 4 still probes 1h–4h, where the effect would actually appear, and tests
`β` rather than correlation — so it is not pre-empted, but the textbook
result should not be assumed either.

**PC2 is not the factor the hazard anticipated.** It was expected to be a
BTC-vs-alt or large-vs-small dimension. It is not: loadings are dominated by
XRP (+0.57) and XLM (+0.57) in ALL20, and XRP (+0.45) / XLM (+0.38) in LONG —
a narrow payment-coin cluster, not a systematic risk factor. A two-coin
idiosyncratic pairing is not something a market-neutral book must hedge; it is
closer to the kind of relative-value structure the strategy might want to
*trade*. This strengthens the single-factor verdict beyond the threshold test.

**Stress confirms the hazard in `BETA_NEUTRAL_DESIGN.md` §8.** PC1 by year:
2020 62.7%, 2021 62.6%, **2022 73.3%**, 2023 62.3%, 2024 58.3%. The factor
share peaks in the crash year — correlations converge under stress, exactly
when neutrality matters most and when a pooled-average check would hide it.
Regime-conditional reporting is not optional.

---

### Step 4 — Beta estimation frequency (Epps check)

**Objective.** Choose the return frequency for estimating `β`.

**Procedure.** Estimate `β` at grid-level (8h), 1h, 2h, 4h, and daily. Report
mean `β̂` and cross-sectional dispersion at each.

**Gate 4.** Adopt the finest frequency at which `β̂` has stabilized — mean
`β̂` changes less than 5% versus the next-coarser frequency.

*Expected:* the finest (highest-frequency) betas biased **low** (Epps effect
from non-synchronous trading), worst for the least-liquid coins;
stabilization around 1h–4h.

**On failure.** If no stabilization appears, do **not** proceed on a guessed
frequency — the bias is differential across coins and is therefore a
cross-sectional distortion, not a harmless level effect.

#### Result — PASSED 2026-08-05: **estimate β at 1h**

`scripts/beta_frequency.py` → `grid/beta_frequency.json`. Each coin is its own
control; only the sampling frequency changes. Window 2021-01-01 → 2024-12-27.

| freq | mean β | std β | mean R² | obs/coin |
|---|---|---|---|---|
| 1h | 1.1342 | 0.170 | 0.568 | 33,900 |
| 2h | 1.1267 | 0.164 | 0.565 | 16,949 |
| 4h | 1.1272 | 0.167 | 0.568 | 8,473 |
| 8h | 1.1187 | 0.155 | 0.553 | 4,235 |
| 24h | 1.0957 | 0.139 | 0.503 | 1,410 |

**There is no Epps effect here.** Every adjacent step moves <2.1%, well inside
the 5% bar, so β is stable across the whole tested range rather than
stabilising at some boundary. Two confirmations:

- The **liquidity signature is absent and reversed**: corr(β change 1h→24h,
  liquidity rank) = **+0.301**. Classic Epps requires a *negative* correlation
  (illiquid coins biased most at high frequency). Most coins' β *falls*
  slightly as frequency coarsens — the opposite direction entirely.
- **Exploratory, beyond the pre-registered set:** even at 5m, mean β is within
  **1%** of the 1h value (5m 1.1229, 15m 1.1369, 30m 1.1349, 60m 1.1342). These
  are the most liquid perps on Binance and trade continuously, so
  non-synchronous trading is a non-issue. 1h has enormous margin.

Consistent with Step 3, where mean pairwise correlation also *fell* with
horizon. The textbook effect does not apply to this universe.

**Why 1h rather than 8h, when the target uses 8h returns.** β is nearly
frequency-invariant here, so the choice is governed by estimation noise, and
that is not close:

| rolling window | SE(β) at 1h | SE(β) at 8h | |
|---|---|---|---|
| 30d | 0.0368 | 0.1059 | 8h is **2.9×** noisier |
| 60d | 0.0260 | 0.0749 | 2.9× |
| 90d | 0.0213 | 0.0612 | 2.9× |

Using a 1h β on 8h returns costs a systematic **+0.0155 (+1.38%)** horizon
mismatch. It buys a noise reduction of **0.0691** at a 30d window — **4.5×
larger than the bias it introduces.**

This matters specifically because of the §7 hazard: the label carries
`(β_true − β̂)·r_m`, an error that scales with market direction. Cutting SE(β)
by ~3× cuts that leak by ~3×. Trading a 1.4% known bias for a 3× noise
reduction is the right side of that exchange.

**Arithmetic consistency check:** volume-weighted mean β is 0.973–0.979 at
every frequency (≈1 by construction, since `r_m` is a weighted average of the
same coins), confirming index and betas are mutually consistent.

---

### Step 5 — Beta window and shrinkage

**Objective.** Fix the estimation window and shrinkage intensity.

**Procedure.** Candidate windows on economic grounds (e.g. 30d / 60d / 90d).

**Gate 5 — selection criterion, stated to avoid circularity.** Choose the
window and shrinkage by **how well `β̂` predicts future realized beta** —
never by strategy Sharpe or backtest performance. Selecting an estimator by
downstream P&L fits the estimator to the backtest and is exactly the
multiple-testing trap this project has already paid for.

*Expected:* best window in the 30–90d range; shrinkage materially improves
out-of-sample beta prediction, confirming raw rolling betas are noisy.

**On failure.** If shrinkage does not help, that is informative — report it
and record the reasoning rather than discarding the result.

#### Result — PASSED 2026-08-05: **30d window of 1h returns, λ=0.7 shrinkage**

`scripts/beta_window.py` → `grid/beta_window.json`. Inputs built once by
`scripts/build_returns_1h.py` (718,194 hourly returns, index vol 71.6%
annualised — consistent with the 8h index's 69.1%).

RMSE predicting realised beta over the following 30 days. No strategy return
enters the script at any point.

| window | none | λ=0.9 | λ=0.8 | **λ=0.7** | λ=0.6 | Vasicek |
|---|---|---|---|---|---|---|
| 15d | 0.2459 | 0.2362 | 0.2296 | 0.2264 | 0.2269 | 0.2396 |
| **30d** | 0.2322 | 0.2250 | 0.2208 | **0.2198** | 0.2219 | 0.2295 |
| 60d | 0.2323 | 0.2271 | 0.2245 | 0.2245 | 0.2273 | 0.2310 |
| 90d | 0.2300 | 0.2262 | 0.2249 | 0.2260 | 0.2295 | 0.2291 |
| 120d | 0.2260 | 0.2232 | 0.2227 | 0.2245 | 0.2285 | 0.2255 |
| 180d | 0.2207 | **0.2190** | 0.2195 | 0.2221 | 0.2267 | 0.2204 |

**Shrinkage works, and the optimal intensity behaves as theory predicts.** At
30d it cuts RMSE from 0.2322 to 0.2198 — **5.34%**. Shorter windows want
heavier shrinkage (15d prefers λ=0.7, 180d prefers λ=0.9), exactly as expected
when the raw estimate is noisier. Vasicek never beats the best fixed λ.

> ### ⚠ SUPERSEDED BY STEP 9 — the target uses **λ = 1.0** (no shrinkage)
>
> The 30d window stands. **The shrinkage does not.** Gate 9B showed that
> λ=0.7 injects exactly the beta-in-disguise contamination the whole design
> exists to prevent, and it was removed. See the Step 9 result for the
> mechanism and evidence.
>
> The analysis above is not wrong — it correctly answers *"which estimator
> best predicts future β."* That was the wrong question for this use.
> **For neutralisation, bias and variance are not interchangeable:** shrinkage
> buys lower MSE by accepting a systematic per-coin bias, and that bias,
> multiplied by `r_m`, becomes a regime-flipping market exposure in the label.
> An unbiased-but-noisier estimator only adds noise. Minimising MSE trades
> exactly the wrong way here.
>
> Shrinkage remains valid for **book construction**, where it is also
> harmless: with `Σw = 0`, imposing `Σwβ̂ = 0` for `β̂ = λβ + (1−λ)β̄` gives
> `λΣwβ = 0`, which is *equivalent* to `Σwβ = 0`. The constant term drops out,
> so shrinkage never affected the neutrality constraint either way.

##### The pre-registered expectation was NOT met — recorded rather than hidden

The expectation was "best window in the 30–90d range". The raw argmin over
the 42-cell grid is **180d + λ=0.9**, outside it.

The deviation rule below was written **after** seeing that result. It is a
judgement made now, in the open, not something pre-registered — stating it
plainly is the point:

> Leaving the pre-registered range requires a *materially* better estimator
> (>5%), not a marginally better one, and the advantage must be stable year to
> year.

180d fails both conditions:

- **The edge is 0.34%** (0.2190 vs 0.2198). Selecting the argmin of 42 cells
  whose total spread is ~7% is noise-mining, not measurement.
- **180d is the worst window in 2022** — the year beta actually moved (LUNA,
  FTX). A slow window cannot adapt to a regime shift, and averaging over five
  years hides that.

RMSE by year, each window at its own best shrinkage:

| year | 15d | 30d | 60d | 90d | 120d | 180d |
|---|---|---|---|---|---|---|
| 2021 | 0.2073 | 0.1959 | 0.1978 | 0.1947 | 0.1926 | **0.1903** |
| 2022 | **0.1958** | 0.1979 | 0.2068 | 0.2053 | 0.2071 | 0.2117 |
| 2023 | 0.2613 | **0.2470** | 0.2578 | 0.2681 | 0.2694 | 0.2571 |
| 2024 | 0.2172 | 0.2137 | **0.2059** | 0.2094 | 0.2093 | 0.2085 |

Best window by year: 2020→120d, 2021→180d, 2022→15d, 2023→30d, 2024→60d.
**No window wins consistently** — the honest reading is that within 30–180d
the choice barely matters, so it should be made on robustness rather than on a
third-decimal RMSE win. 30d is chosen for adaptivity, the cheapest warm-up
cost in training data, and the highest correlation with realised beta (0.643).

##### Two corrections to earlier reporting

- **Shrinkage gain was initially understated as +0.74%.** That compared
  best-shrunk against best-*unshrunk across different windows*, conflating two
  effects. Measured correctly at a fixed window it is **+5.34%**, which
  matches the pre-registered expectation of a material improvement.
- **Residual-loading differences between windows are within noise.**
  Residualising the 8h forward return and regressing on `r_m` gives slopes of
  −0.0096 (15d) to +0.0098 (180d), but the slope's standard error is ≈0.0036,
  so these are ~1 SE apart. Every window sits 5–25× below Gate 9A's 0.05
  threshold. This metric cannot discriminate between windows and was not used
  to select one.

---

### Step 6 — Idiosyncratic volatility

**Objective.** Estimate `σ̂_ε,i,t_obs` from regression residuals over the
Step 5 window.

**Gate 6.**
- `σ̂_ε < σ̂_total` for every coin and period (a strict identity — assert it)
- No zero or degenerate values
- Report the `σ_ε / σ_total` distribution

*Expected:* `σ_ε / σ_total` ≈ 0.4–0.7. Near 1.0 means beta explains nothing
and Step 3 should be revisited; near 0 means coins are pure beta and there is
no cross-sectional opportunity to trade.

#### Result — PASSED 2026-08-05

`scripts/build_beta_idiovol.py` → `grid/beta_idiovol_8h.parquet` (the artifact
Step 8 consumes). Rolling-regression machinery factored into `mft/rolling.py`,
which returns **centered** moments so residual variance can be evaluated for
any β, not only the OLS one — necessary because the target uses a *shrunk* β.

**88,494 rows · 20 coins · 4,597 cross-sections · median σ_ε/σ_total = 0.621**,
inside the expected band. Beta explains ~61% of variance at the median coin.

Idiosyncratic share is economically ordered, which is a good sign the
construction is sane:

| | idio share | note |
|---|---|---|
| BTCUSDT | 0.314 | lowest — BTC largely *is* the market |
| ETHUSDT | 0.352 | |
| SOLUSDT | 0.611 | |
| SUIUSDT | 0.749 | newest listing |
| TRXUSDT | 0.783 | highest — famously idiosyncratic, β 0.78 |

##### Three things measured rather than assumed

**1. The `σ_ε ≤ σ_total` identity holds for OLS but not for shrinkage.** It is
a mathematical guarantee only at the OLS β; a shrunk β can leave more variance
than it removes when the raw β sits far from the cross-sectional mean.
Verified: **0 violations** at OLS, and **442 rows (0.499%)** with the shrunk β.
Real behaviour of shrinkage, not an error — measured instead of assumed away.

**2. The √8 scaling is validated, not taken on faith.** `σ_ε` is estimated at
1h (720 observations per 30d window) and scaled to 8h, rather than estimated
directly at 8h (90 observations). Against a direct 8h estimate: **median ratio
1.054, correlation 0.966.** The 5% overstatement implies mild mean reversion at
hourly scale — 8h variance is slightly below 8× the 1h variance. Small, and it
is a near-constant factor across coins, so cross-sectional ranking is
unaffected.

**3. Self-inclusion in a concentrated index — quantified.** Because `r_m` is
51% BTC, regressing BTC on `r_m` is partly regressing BTC on itself. Measured
against a leave-one-out index:

| coin | weight | β vs `r_m` | β vs `r_m(−i)` | shift |
|---|---|---|---|---|
| BTCUSDT | 0.508 | 0.854 | 0.656 | **−0.198** |
| ETHUSDT | 0.242 | 1.022 | 0.976 | −0.047 |
| SOLUSDT | 0.047 | 1.385 | 1.352 | −0.034 |
| TRXUSDT | 0.005 | 0.640 | 0.635 | −0.005 |

The shift scales cleanly with weight, exactly as mechanical self-inclusion
predicts.

**The full-index β is nevertheless the correct one to use here, and
deliberately so.** The book must be neutral to `r_m` *as actually defined*, and
that requires `β_i = Cov(r_i, r_m)/Var(r_m)` with coin `i` included — setting
`Σ w_i β_i = 0` then makes book beta exactly zero against the real index.
Leave-one-out betas would produce a constraint that neutralises against an
index nobody holds. Consistency then requires the target to residualise with
the same β. Recorded because BTC's "idiosyncratic" return is a slightly odd
object when BTC is half the market.

##### Two findings carried forward

**Idiosyncratic share collapsed in 2022** — median ratio by year: 2020 0.708,
2021 0.610, **2022 0.567**, 2023 0.643, 2024 0.645. This is the
"correlations converge in stress" hazard from `BETA_NEUTRAL_DESIGN.md` §8,
now measured rather than hypothesised: cross-sectional opportunity was
*smallest* in the most stressed year. Any evaluation must be regime-split, and
a flat-in-2022 result should not be read as failure of the signal.

**Weighted residuals sum to ~zero by construction.** Since `Σ w_i r_i = r_m`
and `Σ w_i β_i = 1` at OLS, it follows that `Σ w_i ε_i = 0` exactly (and
approximately under shrinkage). The residual cross-section therefore carries
one linear dependency — with BTC at 51%, BTC's residual is close to the
negative of the weighted sum of all others. Harmless for *ranking*, but it
constrains **book construction**: the 20 residuals are not 20 independent
bets. Flagged for the construction step (out of scope for this document).

---

### Step 7 — Forward returns at h = 8h, from t_fill

**Objective.** Build the label's forward-return leg, starting at `t_fill`,
not at `t_obs`.

**Gate 7.**
- The forward return uses only prices in `(t_fill, t_fill+8h]` — assert
- No forward window crosses the train boundary (purge = `h + lag` = 9h,
  from `t_obs`)
- Coverage ≈ 100% except the final 9h

**On failure.** Any leak here invalidates everything downstream.

#### Result — PASSED 2026-08-05

`scripts/build_forward_returns.py` → `grid/forward_8h_lag60.parquet`.
**89,518 rows · 4,648 cross-sections · 2020-09-28 → 2024-12-31.**

- `fwd_ret` std 0.03226 → **106.8%** annualised (single-coin volatility)
- `fwd_rm` std 0.02074 → **68.6%** annualised, matching Step 2's 69.1%
- Purge **9.00h** from `t_obs` (`h` + `lag`), coverage 99.337% of usable instants

Four properties asserted, not assumed:

- **Entry price closed strictly before `t_fill`** — max age 0.0 min, so no
  candle straddling the fill instant contributes to the entry.
- **The whole label window resolves before the holdout wall**, last close
  2024-12-31 17:00.
- **The forward index equals the Step 2 index shifted one step**, max absolute
  difference **2.78e-17** across 4,648 steps. Two independent constructions of
  the same quantity agreeing to machine precision — the strongest available
  evidence that the weighting, lag and window conventions are mutually
  consistent between Steps 2 and 7.
- **Gaps are never booked as 8h returns** (same consecutive-step rule as Step 2).

The lag sweep {5, 15, 30, 60, 120} min was built in the same pass and written
to `grid/forward_8h_lagsweep.parquet` for Step 7L.

---

### Step 7L — Lag sensitivity check (§3, Gate 3L)

**Objective.** Measure how the target's predictive relationship decays with
`lag ∈ {5, 15, 30, 60, 120}` minutes, everything else fixed.

**Outcome.** Not a strict pass/fail gate — a measurement that determines
whether `lag = 60min` (the default) is viable, and whether faster execution
is worth investing in later. See §3 for the full rationale.

**On a sharp decay (edge largely gone by 60min).** Treat this as a material
finding: the effect is likely microstructural and may not be capturable at
this pipeline's realistic speed. Do not lower `lag` to make the number look
better — that reintroduces the uncapturable-edge problem this step exists to
prevent.

#### Result — MEASURED 2026-08-05: `lag = 60min` retained, with a caveat

`scripts/lag_sensitivity.py` → `grid/lag_sensitivity.json`. No model exists
yet, so a genuine "does the edge survive" test is unavailable and was not
faked. Two measurable things bound the answer instead.

**1. Label drift (model-free).** How much the target itself changes with lag,
against the baseline that two windows of length `T` overlapping by `O` correlate
`O/T` under serially independent returns:

| lag | corr vs lag 5 | IID baseline | difference |
|---|---|---|---|
| 15m | 0.9671 | 0.9792 | −0.0121 |
| 30m | 0.9268 | 0.9479 | −0.0211 |
| **60m** | **0.8586** | **0.8854** | **−0.0268** |
| 120m | 0.7386 | 0.7604 | −0.0218 |

The label degrades slightly *faster* than window overlap alone — consistent
with the mild mean reversion already measured in Step 6 (√8 scaling overstated
8h vol by 5.4%). The effect is small: most of the change is simply the window
being shifted, not information being destroyed.

**2. Probe signal.** Cross-sectional short-horizon reversal — computable from
prices alone, **zero fitted parameters**, so its IC needs no overfitting
correction and in-sample equals out-of-sample.

| lag | IC | block-bootstrap 95% CI | t | retained |
|---|---|---|---|---|
| 5m | +0.0243 | [+0.0151, +0.0336] | 5.14 | 100% |
| 15m | +0.0209 | | | 86% |
| 30m | +0.0167 | | | 69% |
| **60m** | **+0.0133** | **[+0.0045, +0.0224]** | **2.91** | **55%** |
| 120m | +0.0132 | | | 54% |

A real cross-sectional reversal effect exists in this universe and **survives
60 minutes of delay** with a CI excluding zero.

##### The shape of the decay is the useful finding

Decay is **not linear**. It falls steeply from 5→60min (100% → 55%) and then
goes flat: 60→120min loses ~1%. That is the signature of two components — a
fast one dying within the hour, and a slower one that further delay barely
touches.

Operationally:

- **Being slower than 60min costs almost nothing.** If the pipeline slips to
  120min, ~1% of the probe effect is lost. There is no cliff to the right.
- **Being faster is worth real money.** Reaching 15min would recover 86% from
  55% — a **57% relative improvement** in probe IC. The cliff is entirely
  between 5 and 60 minutes.

That gives a quantified engineering target if execution speed ever competes for
effort against feature work.

##### Two things on the record

- **Naive t-stats are approximately valid here**, unlike the previous project
  where overlapping labels made them ~4× overconfident. Per-bar IC
  autocorrelation is only **+0.039**, and block bootstrap moves t from 5.88 to
  5.14 (lag 5) and 3.25 to 2.91 (lag 60). This is a structural benefit of
  setting the label horizon equal to the rebalance interval: consecutive labels
  do not overlap.
- **The probe is reversal, among the fastest-decaying effects known.** A model
  built on slower mechanisms (carry, positioning) would decay *less*, so 55%
  retention is likely a **pessimistic bound** rather than an estimate of what a
  real model keeps.

**Decision: `lag = 60min` retained** as the conservative default, matching the
measured 50–65min pipeline latency. It costs materially against a hypothetical
instant fill, but that fill was never achievable, and the effect that survives
is statistically real.

---

### Step 8 — Assemble the target

**Objective.** Combine Steps 2, 5, 6, 7 into `t_i,t_obs`.

**Gate 8 — reconstruction identity. This gate ABORTS; it does not print.**
Independently recompute the target from its stored components and require
`corr(recomputed, stored) ≥ 0.999` **and** `max|difference| < tol`.

*Expected:* `corr = 1.000000`.

This gate exists because a target that is nominally correct but subtly
mis-assembled produces plausible-looking results that are wrong by
multiples. A diagnostic that is printed and not read is not a diagnostic.

#### Result — FAILED, then PASSED 2026-08-05

`scripts/build_target.py` → `grid/target_8h_lag60.parquet`.
**87,860 rows · 20 coins · 4,558 cross-sections · 2020-10-28 → 2024-12-31.**

The gate was implemented at two depths, because they catch different things:

- **8A — algebraic, every row.** Recompute the target from stored components.
  Catches assembly and join errors.
- **8B — independent rebuild from 1m klines, sampled instants.** Rebuild
  *everything* — prices, weights, the 1h index, rolling betas, shrinkage,
  idiosyncratic vol — bypassing every intermediate parquet. **8A cannot see a
  stale artifact or a drifted convention, because it trusts the same inputs
  that produced the error.** Re-running the same code path and getting the same
  answer proves nothing; only an independent derivation is evidence.

**8B failed on the first run, and the abort was correct.** 8A was perfect
(`corr 1.00000000`, `max|diff| 0.00e+00`) while 8B showed `sigma_eps`
disagreeing by up to **94%**. Had the gate merely printed, that would have been
easy to skim past — a 94% error in the target's denominator.

**Diagnosis.** 464 of 480 sampled rows agreed to ~1e-15. All 16 disagreements
sat at a single instant, 2020-10-20 — the earliest sampled date. Beta is
regressed on the market index, but the index only exists once volume weights
do (2020-09-28, after their own 30d warm-up). Targets dated before
`weights_start + beta_window` therefore have betas fitted against an index that
did not exist for part of their window: **median `n_obs` 572 of 720 there,
versus 720 after.** An independent rebuild cannot reproduce those, because
reproducing them depends on exactly when the index blinked into existence.

**Fix — a warm-up trim, added in response to the gate and NOT pre-registered.**
The target now starts at `weights_start + 30d` = 2020-10-28, dropping **576
rows (0.65%)**. This removes a genuine soft spot rather than silencing a check:
those betas rest on a shorter, differently-composed sample regardless of
whether anyone verifies them. Recorded as post-hoc because it is.

**After the fix, both reconstructions are exact:**

| check | corr | max diff |
|---|---|---|
| 8A algebraic (all 87,860 rows) | 1.00000000 | 0.00e+00 |
| 8B independent rebuild (480 rows, 25 instants) | 1.00000000 | 2.20e-14 |
| — component `beta` | | 5.67e-16 |
| — component `sigma_eps` | | 3.29e-15 |
| — component `fwd_rm` | | 3.74e-16 |

##### Target distribution — and one finding for Gate 9C

| | |
|---|---|
| mean | +0.0030 |
| std | 1.0984 |
| **skew** | **+2.901** |
| **excess kurtosis** | **+67.51** |
| \|target\| > 5 | 0.429% |
| \|target\| > 10 | 0.0524% |

Std near 1 confirms the standardisation works — `σ_ε` estimated on a trailing
window does scale the forward residual to roughly unit variance.

**The tails are very heavy.** Kurtosis of 67 on a standardised quantity means
frequent extreme values, and skew +2.9 means a long right tail — individual
coins occasionally making outsized idiosyncratic moves. This makes the
winsorisation question in Gate 9C a live one rather than a formality.

Worth noting it also **retroactively supports the ranking objective**: a
squared-error loss would let those 0.4% of rows beyond ±5 dominate the fit,
whereas ranks are bounded and largely indifferent to tail magnitude.

Cross-sectional dispersion is stable year to year (0.86–0.93), so the heavy
tails are a property of the pooled distribution, not of a particular period.

---

### Step 9 — Target validation

**Objective.** Establish that the target is actually what it claims to be.

**Gate 9A — residual market loading.** Regress `t_i` on `r_m,[t_fill,t_fill+h]`
(pooled). The slope must be ≈ 0. *Expected `|slope| < 0.05`.* This is the
direct test of whether residualization worked.

**Gate 9B — regime split on the target itself.** Compare target behaviour in
up-market versus down-market bars. A systematic difference means beta
estimation error is leaking market direction into the label (§7).
*Expected: difference small relative to bootstrap noise.*

**Gate 9C — distribution.** Cross-sectional dispersion, tails, no explosive
values. Any winsorization decision is deferred and must be made explicitly,
as its own one-variable experiment.

**Gate 9D — noise floor.** Bootstrap the cross-sectional dispersion
**before** quoting any point estimate downstream. Intervals, not points.

**On failure of 9A or 9B.** Return to Steps 4–5. Do not proceed to modelling
with a target that carries market loading.

#### Result — FAILED, then PASSED 2026-08-05, after a real change to the target

`scripts/validate_target.py` → `grid/target_validation.json`.

**Step 9 failed on the first run with 5 of 10 gates red.** Two separate things
were wrong, and they are kept apart deliberately: one was a **genuine defect in
the target**, the other was **three thresholds I had mis-specified**. Conflating
them would make a real fix look like moved goalposts.

##### The genuine defect: shrinkage was injecting beta-in-disguise

Gate 9B measures `corr(target, β)` within each bar, separately in up- and
down-market bars. If beta removal leaves exposure, that correlation flips sign
with market direction. It did:

| | λ = 0.7 (before) | λ = 1.0 (after) |
|---|---|---|
| corr(target, β) — up bars | +0.0631 | −0.0144 |
| corr(target, β) — down bars | −0.0989 | −0.0152 |
| **gap** | **+0.1619** | **+0.0008** |
| corr(target, σ_ε) gap | +0.1339 | +0.0238 |
| unscaled residual slope, worst coin | −0.107 (TRX) | −0.007 pooled |

**Mechanism, verified not assumed.** Per-coin residual loading was regressed
against the shrinkage push `(β_shrunk − β_raw)`:
**corr = 0.80.** BTC is near-exact — shrinkage pushed its β up by +0.077 and
left a −0.0772 residual loading. TRX: pushed +0.142, left −0.107. Low-β coins
were over-subtracted, high-β coins under-subtracted, precisely as
`−(1−λ)(β_i − β̄)` predicts.

**Fix: λ = 1.0.** Raw OLS β is unbiased, so the label carries only zero-mean
noise instead of a systematic per-coin market exposure. Costs ~5.6% on β
prediction RMSE (Step 5) and buys the elimination of the contamination the
entire beta-neutral design exists to prevent.

##### The mis-specified thresholds — provably, not conveniently

Three gates I wrote were wrong in units. The proof is independent of the
result:

**`|slope| < 0.05` on the scaled target was unachievable by construction.**
Because `slope = corr × std(target)/std(r_m) = corr × 53`, passing it requires
`|corr| < 0.00094`. The standard error of a correlation at n = 87,860 is
`1/√n = 0.0034` — **3.6× larger than the threshold**. A *perfectly* neutral
target would fail this gate about **78% of the time by chance**. The threshold
was borrowed from return-space intuition and applied to a σ-scaled quantity.

Replaced with the scale-free test — is the correlation distinguishable from
zero? **corr = −0.00389, SE = 0.00337, t = −1.15. Not significant.**

The other two, same character: the per-coin gate is now a Bonferroni-corrected
t-test over 20 simultaneous tests (worst |t| = 2.71, SUI, against 3.02), and
the dispersion-CV gate was dropped — a raw CV on a positive right-skewed
quantity is not meaningful, and **under a ranking objective per-bar scale is
irrelevant anyway**, since ranks are invariant to it. It now checks only that
dispersion never collapses, which would make a bar unrankable (min 0.206).

##### Final state — all 10 gates pass

| gate | result |
|---|---|
| 9A scale-free market correlation | corr −0.00389, t **−1.15** |
| 9A unscaled residual slope | **−0.0071** |
| 9A per-coin, Bonferroni | worst \|t\| **2.71** < 3.02 |
| **9B beta regime gap** | **+0.0008 ± 0.0188** |
| 9B σ_ε regime gap | +0.0238 ± 0.0194 |
| 9C tails / dispersion / concentration | max \|t\| 47.8, min dispersion 0.206, largest coin share 8.5% |
| **9D IC noise floor** | **SE = 0.00366** |
| 9D target autocorrelation | max \|ρ\| 0.0911 |

##### The number to remember: the noise floor

**IC standard error under the null is 0.00366** (analytic 0.00347, Monte Carlo
over 200 random signals 0.00366 — cross-checked). Therefore:

- **\|IC\| > 0.0072** to clear 2σ
- **\|IC\| > 0.0094** to clear 3σ

This is established **before any model exists**, so no future IC can be quoted
without a scale to judge it against. For context, the Step 7L reversal probe
scored IC +0.0133 at lag 60 — roughly 3.6σ.

Winsorisation remains **deliberately not applied**: the ranking objective is
bounded, so tail magnitude cannot dominate the fit. If it is ever tested it
must be its own one-variable experiment.

---

### Step 10 — Does residualization earn its keep?

**Objective.** Settle open question #5 of the base design. This step
requires a model and is the handoff point into modelling work; the decision
rule is pre-registered **now**, before any result exists.

**Arms — exactly one variable differs.**
- **A:** `(r_i − β̂_i·r_m) / σ̂_ε,i` (as specified)
- **B:** `r_i / σ̂_i` (vol-normalized raw return, no residualization)

Identical features, identical construction, identical seeds.

**Gate 10.** Compare on **out-of-sample-within-train** walk-forward folds,
reporting all folds together — never the best. Judge on ex-post book beta
and paired per-seed IC, with intervals.

*Expected:* A produces materially lower ex-post book beta and requires
smaller deviation from intended weights in the neutrality projection. If A
is not better on beta, its extra estimation noise is not being repaid and B
is preferable.

*Prior recorded in advance:* A wins on beta control. The argument is that
ranking on total return puts high-beta coins on top in up-markets, forcing
the neutrality projection to fight the signal — whereas a residualized
ranking is already approximately beta-balanced, so the projection is a small
correction. This also bears on open question #6 of the base design (how much
freedom the projection has).

---

## 9. Open items

1. Beta estimation frequency — resolved by Gate 4
2. Beta window and shrinkage intensity — resolved by Gate 5
3. Winsorization of the target — deferred; its own one-variable experiment
4. Whether `h = 8h` survives contact with realized turnover and slippage —
   revisit only with evidence, and as a deliberate change; 24h is the
   documented fallback
5. Two-factor form of the target, if Gate 3 triggers it
6. Whether `lag = 60min` is the right default, or should move once Step 7L
   results exist
7. Minimum cross-section floor (candidate: 15) — confirm against actual gap
   frequency once Step 0 runs
8. **Asymmetric beta — a gap Gate 9B does not cover.** 9B tested whether
   `corr(target, β)` flips sign between up and down markets (a *linear*
   leakage test) and it passed cleanly, gap +0.0008. It did NOT test whether
   the residual carries *asymmetric* loading. If `β⁻` differs materially from
   `β⁺`, then `r − β·r_m` leaves a directional component the symmetric test
   cannot see: the residual would run systematically negative in down markets
   and positive in up markets for high-`β⁻` coins, while its correlation with
   β stays flat. Estimate β separately on up- and down-market 1h returns and
   compare. Cheap — `mft/rolling.py` already returns the centered moments
   needed. Worth settling before the feature programme, since a finding here
   changes the target rather than the feature list.
   See `docs/FEATURE_EXPLORATION.md` §15.4.

   > **SETTLED 2026-08-28 — Gate 9C, `scripts/gate_semibeta.py`. It FAILS, and
   > the failure is real, but it is not the failure this item predicted.**
   >
   > All four sub-gates fail: `corr⁻` +0.0258 (t +5.29), `corr⁺` −0.0593
   > (t −12.73), difference +0.0851 (t +12.61), and 11 of 20 coins are
   > individually significant after Bonferroni.
   >
   > But the mechanism is **not** directional. Linear neutrality holds exactly
   > as 9A found — `corr(target, r_m)` = **−0.0039, t −1.15**. What the target
   > carries is **convexity**: `corr(target, |r_m|)` = **−0.0435, t −12.91**.
   > The target runs systematically negative when the market moves hard *in
   > either direction*, which is why splitting by sign produced two opposite
   > slopes and looked like asymmetry. A single β per coin cannot represent a
   > loading that depends on `|r_m|`, so the residual keeps one. BTC carries
   > the opposite sign (+0.0548) because it dominates the index weight.
   >
   > **What this does and does not threaten.** The common part cannot reach a
   > cross-sectional ranking objective, which consumes only within-bar order.
   > (The measured `corr(within-bar demeaned target, |r_m|)` is exactly 0.00000
   > — but that is a **tautology**, since `|r_m|` is constant within a bar and
   > the demeaned target sums to zero within it. It is recorded as such and is
   > not offered as evidence.) What *can* reach a book is the cross-sectional
   > **dispersion** of convexity, which is real: per-coin `corr(target,|r_m|)`
   > has mean −0.0417 and **std 0.0440**, ranging −0.112 (TRXUSDT) to +0.070
   > (BTCUSDT). Its correlation with mean β is +0.359 (t +1.63, n=20 —
   > underpowered, not evidence of absence).
   >
   > **The thresholds were not relaxed to make this pass.** The gate stands as
   > failed. What changes is the *claim*: the design is neutral to the **linear**
   > market factor, cross-sectionally, and is **not** neutral to market
   > convexity at the coin level. Anywhere this project says "beta-neutral", it
   > means the former.
   >
   > **Carried forward as a portfolio-level check, not a target change.** A
   > convexity term (`β_i·r_m + γ_i·r_m²`) would be a target redesign, and the
   > brief is the base beta-neutral case. Instead: once a book exists, measure
   > its realised `|r_m|` exposure directly. Registered in
   > `FEATURE_SELECTION.md` §8.

---

## 10. Method discipline

`docs/BETA_NEUTRAL_DESIGN.md` §10 applies in full. The rules most
load-bearing here:

- **One variable per arm** (Step 10 is written to this standard)
- **Gates abort, they do not print** (Gate 8 especially)
- **Pre-registered decision rules with expected results**, so nothing can be
  retrofitted — every gate above states its expectation in advance
- **Estimators are never selected on downstream P&L** (Gate 5)
- **Regime split** on anything claiming neutrality — including the target
  itself
- **Noise floor before point estimates**

All research runs on train only (2020-01-01 → 2024-12-27). 2025 and 2026
remain sealed. See `mft/splits.py`. The purge for this target is `h + lag`
(9h at current defaults), not the legacy 120h.
