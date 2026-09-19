# Feature exploration — public Binance data

*Drafted 2026-08-06. **Status: EXPLORATION, not a commitment.** This is the
idea pool — the superset from which the frozen list was drawn.*

> **The frozen list now exists: `docs/FEATURE_LIST_FROZEN.md`** (31 features,
> 6 families, frozen 2026-08-07). This document remains the pool and may keep
> growing; the frozen list may not. Ideas added here after that date are
> candidates for a future amendment, not part of the pre-registered set.

Keeping those two things apart is deliberate. Exploring freely and then
pre-registering a frozen list before any code touches the data is what stops
the search itself from becoming the overfitting.

---

## 1. What the target already decided for us

The target is fixed (`docs/TARGET_DESIGN.md`), which turns four open questions
into numbers:

| constraint | value | consequence |
|---|---|---|
| Horizon | 8h, entered after a 60min lag | anything decaying inside an hour is dead on arrival |
| Effective sample | 4,558 cross-sections | ~110 features at 40:1 — and that is the *optimistic* bound |
| Noise floor | IC SE **0.00366** | a feature needs \|IC\| > **0.0072** for 2σ, **0.0094** for 3σ |
| Benchmark | reversal alone scores **IC 0.0133** | **zero is not the bar.** 0.0133 is |

That last row matters more than it looks. A parameter-free reversal signal
already clears 3.6σ. Any feature set has to be judged against *that*, not
against nothing — which is exactly the comparison that would have caught the
previous project's problem early.

### The design principle that follows

**Build features in the same space as the target.**

The target is a *beta-residualised, idio-vol-normalised* return. A feature
carrying market exposure can only predict a component the target no longer
contains, so its beta part is pure dilution. Residual momentum, not raw
momentum. Idiosyncratic vol, not total vol.

`mft/rolling.py` already produces exactly these, so the residualised versions
cost nothing extra.

### The second principle: prefer surprises to levels

**Most raw quantities are dominated by permanent coin characteristics.** BTC
always has more volume than ZEC. Under a ranking objective, a permanent
cross-sectional difference that does not predict is **pure noise consuming a
feature slot** — the model ranks the same coins the same way every bar for
reasons unrelated to what happens next.

So for most quantities the informative form is the **surprise** — deviation
from that coin's own recent norm — rather than the level:

```
volume_surprise = volume / trailing_mean(volume)
oi_surprise     = ΔOI relative to its own typical ΔOI
vol_surprise    = realised vol / trailing typical vol
```

**How much this matters is measurable, and it varies enormously.** The ratio
`|rolling mean| / rolling std` says what fraction of a quantity is level
rather than variation (measured 2026-08-07, 30d windows):

| quantity | \|rolling mean\| / rolling std | demeaning is… |
|---|---|---|
| **Funding** | **0.904** | **essential** — level ≈ variation |
| Returns | 0.073 | nearly a no-op — already near-zero mean |

Run that ratio on any candidate before deciding whether to demean it. It is
one line and it settles the question.

**The caveat that stops this being a rule:** some permanent characteristics
genuinely *do* predict — illiquid coins really do revert more, and `σ_ε/σ_total`
(§8.2) is a persistent characteristic that is a legitimate feature. Level and
deviation carry different information, and a few quantities want both. The
*default* should be the deviation.

---

## 2. Data inventory — what is actually on disk

| dataset | resolution | fields |
|---|---|---|
| `perp_klines` | 1m | OHLC, volume, quote_volume, **trade_count**, **taker_buy_base/quote_volume** |
| `spot_klines` | 1m | same fields, spot market |
| `premium_index` | 1m | OHLC of the perp-vs-mark premium |
| `funding` | 8h | `last_funding_rate`, `funding_interval_hours` |
| `metrics` | 5m | `sum_open_interest`, `sum_open_interest_value`, `count_toptrader_long_short_ratio`, `sum_toptrader_long_short_ratio`, `count_long_short_ratio`, `sum_taker_long_short_vol_ratio` |
| `um_delivery_klines` | 1m | quarterly futures (BTC/ETH only — market context, not cross-sectional) |

**NOT downloaded — needs a backfill run before use:**

- **Liquidation snapshots** (`futures/um/daily/liquidationSnapshot`). A spec
  already exists in `mft/data/specs.py` (`LIQUIDATION_COLUMNS`) but no data has
  been fetched.
- **aggTrades** — individual prints. Available but very large.

The headline: **two parallel 1-minute tapes for every coin (spot and perp),
plus a dislocation series, plus 5-minute positioning.** Richer than it looks,
and mostly unexploited.

---

## Seam 1 — Carry and funding

**The most horizon-matched mechanism available.** Funding settles every 8h;
the decision grid is every 8h; each window contains exactly one settlement.

A structural bonus falls out of a decision made for other reasons: because
funding was **excluded from the target**, a funding feature is a clean test of
one hypothesis — *does crowded positioning predict price reversal?* It cannot
be contaminated by mechanically earning the carry, because the carry is not in
the label.

| idea | construction | story |
|---|---|---|
| **Level** | annualised funding, cross-sectionally ranked | baseline crowding measure |
| **Clamp detection** | is funding pinned at the ±cap? | demand so extreme the mechanism could not fully price it — a distinct state the level alone hides |
| **Funding surprise** | realised funding − premium-index-implied funding | funding is derived from the premium plus an interest term, with clamping; the gap is the surprise |
| **Cumulative burden** | Σ funding paid over trailing 7d / 30d | not "is it expensive now" but "how long have longs been bleeding" — accumulated stretch a spot reading misses |
| **Sign persistence** | consecutive same-sign settlements | entrenched crowding vs a fresh flip |
| **Funding × ΔOI** | funding interacted with OI direction | high funding + growing OI = new crowding; high funding + shrinking OI = capitulation. Opposite meanings, same funding number |
| **Cross-sectional dispersion** | spread of funding across the universe | market-wide crowding regime (bar-level — see §12) |

### 1.1 Crowd-winning — funding surprise × return surprise

```
z_f = ( funding − rolling_mean(funding) ) / rolling_std(funding)
z_r = ( return  − rolling_mean(return)  ) / rolling_std(return)

crowd_winning = z_f × z_r
```

Both legs use only data at or before `t_obs`: funding settled at or before the
decision, returns realised before it. No forward window.

**The four states collapse to one axis**, which is why the product works:

| state | reading |
|---|---|
| funding high + return up | crowded longs **winning** |
| funding low + return down | crowded shorts **winning** |
| funding high + return down | crowded longs **losing** |
| funding low + return up | crowded shorts **losing** |

Positive → the crowded side is winning → **fragile, expect reversal**.
Negative → the crowded side is losing → **squeeze building**.

#### Measured 2026-08-07 — why the construction is shaped this way

| | \|rolling mean\| / rolling std | sign flips per bar |
|---|---|---|
| **Funding** | **0.904** | **29.2%** — a persistent state |
| **Returns** | **0.073** | **52.1%** — a coin flip |

- **Demeaning funding is essential.** Its persistent level is nearly as large
  as its variation, and per-coin mean funding is stable across halves of the
  sample (rank correlation +0.53). Without demeaning, the feature would rank
  coins on a permanent characteristic rather than current state.
- **Demeaning returns is nearly a no-op** — 8h returns already have near-zero
  mean, so the rolling mean is ~7% of the standard deviation. Harmless, but do
  not expect it to do work.
- **The two axes are genuinely independent**: `corr(z_f, z_r) = +0.075`, with
  the four states balanced at 27.4 / 28.4 / 21.8 / 22.4%.

#### Why continuous rather than a 4-state categorical

A sign-merged version (four dummy states) is the obvious alternative and is
worth testing, especially given the target's kurtosis of 67 — discarding
magnitude is robust to tails. But it costs two things:

1. **Magnitude.** Funding pinned at the ±0.75% clamp and funding barely above
   its mean get the same sign. That distinction is exactly why clamp detection
   is a separate row in the table above.
2. **Ranking resolution.** Four states across ~19 coins ties roughly five coins
   per state, and `lambdarank` cannot order within a tie. A continuous product
   preserves the full ordering.

Also note **17.8% of funding observations sit at |z_f| < 0.25**, where the sign
is coin-flip territory. The continuous form degrades gracefully there; the
categorical form assigns a confident state to noise.

Test both as a one-variable experiment. Expected sign recorded in advance:
**negative IC** — crowd-winning predicts reversal.

---

## Seam 2 — Premium index and arbitrage capacity

`premium_index` is a 1-minute OHLC series measuring perp-vs-mark dislocation,
and it is almost entirely unused.

**The idea worth the most here: premium mean-reversion speed.** Fit an AR(1)
to the 1m premium over the trailing window.

- **Fast reversion** → arbitrageurs active and well-capitalised
- **Slow reversion** → dislocations persist because nobody has balance sheet
  to close them

That is a **coin-specific arbitrage-capacity / stress measure**, and it is not
a variant of anything else in this pool.

Cheaper companions from the same series:

- **Premium volatility** — instability of the perp-spot link
- **Time above/below zero** — fraction of the 480 minutes on each side
- **Premium range vs level** — dislocation amplitude relative to its mean
- **Premium asymmetry** — skew of the 1m premium distribution

---

## Seam 3 — Intra-bar structure

Every decision collapses 480 one-minute bars into a single number. The path is
free information that is normally discarded.

| idea | construction | story |
|---|---|---|
| **Path efficiency** | `\|net return\| / Σ\|1m returns\|` | a clean trend and a churned net-zero-then-3% move mean different things |
| **Volume concentration** | Herfindahl of the 480 per-minute volumes | one burst vs steady flow; burst-driven moves revert harder |
| **Timing of the move** | share of the 8h return realised in the first hour vs the last | a move made early *and held* differs from one that just printed |
| **Close vs intra-bar VWAP** | `Σ(close·vol)/Σvol` over the bar, compared to close | closing above your own VWAP = buyers held control into the close |
| **Realized skew / kurtosis** | 3rd/4th moments of the 1m returns | detects a crash-shaped move inside an otherwise ordinary bar |
| **Parkinson vs close-to-close vol** | high-low range vol ÷ realised vol | separates gappy from continuous movement |
| **Volume–return coupling** | within-bar corr(\|1m return\|, 1m volume) | is volume informative for this coin right now, or noise? |
| **Jump share** | `(RV − BV) / RV`, bipower variation vs realised variance | how much of recent movement was *jumps* rather than diffusion. A coin that moved 5% in three minutes is in a different state from one that drifted 5% over eight hours |
| **Signed jump** | sign of the dominant jump component | a downward jump and an upward one of equal size are not the same event |

**On jumps specifically.** Bipower variation (Barndorff-Nielsen–Shephard) is
robust to jumps while realised variance is not, so their difference isolates
the jump component. Computable from 1m returns at no extra data cost. It
overlaps partially with path efficiency — both distinguish "how" a move
happened — but not fully: path efficiency measures directional cleanliness,
jump share measures concentration in time. Worth checking their correlation
before carrying both.

---

## Seam 4 — The retail-attention detector

You have `trade_count` **and** `volume` as separate fields. Their ratio is
average trade size, but the interesting quantity is the divergence of their
*anomalies*:

```
A = trade_count / trailing_mean(trade_count)
B = volume      / trailing_mean(volume)
feature = A - B      (or log A - log B)
```

- **A high, B flat** → many small trades → **retail piling in**
- **B high, A flat** → few large trades → **institutional accumulation**

Same tape, opposite meaning. Economic story is clean and directional:
retail-driven moves should revert, size-driven moves should persist.

**Why this is among the most interesting ideas here:** free, genuinely
cross-sectional, and not a variant of anything else in the pool.

**How it could fail:** average trade size drifts secularly as exchanges change
tick/lot conventions, and it differs enormously across coins by price level
(DOGE at $0.10 vs BTC at $60k mechanically produces different trade counts).
Both are handled by normalising against each coin's *own* trailing history
rather than cross-sectionally in raw units — but that normalisation is doing a
lot of work and should be checked.

---

## Seam 5 — Spot vs perp, two windows on one asset

Unique to crypto. You cannot do this in equities: two continuously traded
venues for the same asset with materially different participant mixes.

| idea | construction | story |
|---|---|---|
| **Leverage intensity** | perp quote_volume ÷ spot quote_volume | high = speculative froth rather than real demand |
| **Aggressor divergence** | taker-buy share in perp − same in spot | leveraged longs buying while spot sells is a fragile configuration |
| **Lead–lag** | cross-correlation of 1m perp vs spot returns at ±k minutes | *which venue discovers price for this coin?* Perp-led moves should be less durable |
| **Return gap** | perp return − spot return over the window | mean-reverting dislocation |
| **Trade-count ratio** | perp vs spot trade_count | different participant mix, independent of size |

**How lead–lag could fail:** it needs enough 1m observations to estimate a
cross-correlation per coin per bar, and the estimate will be noisy at 480
points. It may only be usable on a longer trailing window (e.g. 7d), which
makes it a slow-moving feature rather than a bar-level one.

---

## Seam 6 — Free microstructure, no order book required

| idea | construction | note |
|---|---|---|
| **Zero-return minutes** | count of the 480 minutes with no price change | the crypto analogue of the Lesmond–Ogden–Trzcinka zero-return illiquidity measure. Captures what Amihud misses: Amihud is impact *per unit volume*, this is how often price simply does not move because nothing crossed the spread. Nearly free — and if it is zero everywhere in a liquid 20-coin universe, one query tells you |
| **Roll's implied spread** | `2√(−cov(r_t, r_{t−1}))` on 1m returns | effective spread with no book at all; undefined when the covariance is positive, which is itself informative |
| **Kyle's lambda** | regress 1m returns on signed volume within the bar | genuine per-coin, per-bar price impact — which coin is most impact-sensitive *now* |
| **Amihud illiquidity** | `\|return\| / dollar volume` | classic, cheap, robust |
| **Order-flow autocorrelation** | serial correlation of signed taker volume | persistent flow reads as informed; alternating as noise |
| **Corwin–Schultz spread** | from high-low ratios over adjacent periods | second spread estimator, useful as a cross-check on Roll |

---

## Seam 7 — The positioning fields nobody unpacks

`metrics` carries **four** distinct ratios. The ratios *between* them carry
more than any one alone:

| idea | construction | story |
|---|---|---|
| **Conviction / concentration** | `sum_toptrader_ls ÷ count_toptrader_ls` | are the *biggest* top traders more long than the typical one? |
| **Smart-vs-retail spread** | `sum_toptrader_ls` vs `count_long_short_ratio` | top traders against the broad account base |
| **Flow vs stock** | `sum_taker_long_short_vol` (flow) vs `sum_toptrader_ls` (stock) | flow buying into a short stock = a squeeze building |
| **OI velocity** | ΔOI / OI over 1h / 8h / 24h | position building or unwinding |
| **OI–price quadrant** | rolling `corr(ΔOI, Δprice)` | one continuous feature separating new longs / new shorts / squeezes / capitulation |
| **OI per trade** | ΔOI ÷ trade_count | are positions built by many small or few large participants? |
| **Turnover** | volume ÷ OI | how fast the position base churns |

---

## Seam 8 — Structural, cycle and state

### 8.1 Variance ratio — *where does reversal actually work?*

```
VR(k)  =  Var(k-period return)  /  ( k · Var(1-period return) )
```

computed on **1h residuals**, for k on the ladder (8 / 24 / 72 / 168 hours).

- **VR < 1 → mean-reverting**
- **VR > 1 → trending**
- **VR ≈ 1 → random walk**

**This is the highest-value conditioning variable in the pool, because
reversal is the benchmark signal.** Step 7L measured a parameter-free reversal
probe at IC +0.0133 (t = 2.91). Variance ratio says *which coins are currently
in a state where that should work* — a coin at VR 0.7 and one at VR 1.3 ought
to be treated in opposite directions by the same reversal feature, and nothing
else in the pool distinguishes them.

Two uses, and they are different features:

1. **Standalone** — a coin's current mean-reversion regime, cross-sectionally
   ranked.
2. **Conditioning** — `residual_return × VR`, encoding "reverse harder where
   reversion is actually happening" (see §14.7 on explicit interactions).

**How it could fail.** VR is a ratio of two noisy variance estimates, so it is
noisier than either. It needs enough 1h observations to be stable, which
argues for estimating it on a long window (90d+) even though the signal it
conditions is short. And the choice of `k` is itself another lookback
decision — which the agreement construction in §14.6 partly answers.

### 8.2 Idiosyncratic share — a feature you already compute

`σ_ε / σ_total` is emitted by Step 6 as a diagnostic (`ratio` in
`beta_idiovol_8h.parquet`) and was never listed as a candidate. It should be.
It is **free** — already built, already validated, already point-in-time — and
it spans 0.31 (BTC) to 0.78 (TRX) cross-sectionally.

**It is also the BTC-decoupling measure**, which is not obvious. Measured
2026-08-07 against rolling 30d `corr(coin, BTC)` on raw 1h returns:

| | |
|---|---|
| pooled correlation | **−0.940** |
| **mean within-bar rank correlation** | **−0.955** |

Within any bar, ranking coins by correlation-to-BTC is essentially the reverse
of ranking them by idiosyncratic share. Algebraically unsurprising —
`σ_ε/σ_total = √(1−R²)` against an index that is 51% BTC with
`corr(r_m, r_BTC) = 0.948` — but worth having measured, because **a separate
"rolling correlation to BTC" feature would spend a slot on something already
present.**

The underlying quantity is genuinely dynamic, so this is not a small thing:
cross-sectional spread within a bar averages 0.098, and individual coins swing
hard — DOGE's 30d correlation to BTC ranges from **0.08 to 0.87** across the
sample. Decoupling episodes are real; they are simply already captured.

*Interpretive note:* high idio share means the coin is doing its own thing, so
its residual carries more genuine signal-or-noise; low idio share (BTC, ETH)
means the residual is a small difference between two large numbers and is
correspondingly noisier per unit of return.

#### Horizon-matched BTC correlation — the version that IS additional

The redundancy above is specific to **1h** correlation, because `σ_ε` is
estimated at 1h and scaled by √8. Correlation measured at the **holding
horizon** is a different number: Step 4 found mean R² against the index falling
from 0.568 at 1h to 0.503 at 24h, so unlike beta — which came out
frequency-*invariant* — **correlation is horizon-dependent in this universe.**

Measured 2026-08-07, within-bar rank correlation against idio share:

| measure | redundancy | persistent? |
|---|---|---|
| 1h / 30d correlation | −0.955 | — |
| 8h / 30d correlation | −0.796 | — |
| **8h / 90d correlation** | **−0.696** | **yes: autocorr +0.218, rank persistence +0.411** |
| Sign agreement (8h/30d) | −0.566 | untested |
| ~~Term structure (corr₈ₕ − corr₁ₕ)~~ | ~~+0.295~~ | **NO: +0.044 / +0.038 — noise** |

**`corr₈ₕ(90d)` is a legitimate candidate.** It is persistent, so it measures a
real coin property, and at 70% redundancy roughly a third of it is not already
in the idio share. Use a 90d window: at 8h there are only 90 observations in
30d (SE 0.105) versus 270 in 90d (SE 0.061).

#### And a trap that nearly worked

The correlation **term structure** looked like the best idea in this group —
only +0.295 redundant, apparently almost pure new information. It is noise.
Non-overlapping autocorrelation +0.044 and cross-sectional rank persistence
+0.038: the ordering it produces this month has no relationship to next
month's.

**Low redundancy has two explanations — new information, or noise — and noise
correlates with nothing.** A redundancy check alone cannot tell them apart.
Every low-redundancy candidate must pass a **persistence test** before being
believed: does the ordering it produces survive to a non-overlapping period?

A weak stable *per-coin average* does survive (first-half vs second-half rank
correlation +0.536, n=19, t≈2.3), but that is a near-constant characteristic
producing the same tilt every bar — not a dynamic signal.

*Still untested:* whether sign agreement is persistent, and the interaction of
BTC correlation with funding state (a decoupled coin carrying extreme funding
is a distinct configuration from a coupled one).

### 8.3 Other structural and state features

- **Session.** The 8h grid is exactly 00:00 / 08:00 / 16:00 UTC — Asia /
  Europe / US. Coins have genuinely different session personalities, and this
  costs nothing.
- **Session-relative volume.** Compare a coin's volume to *its own typical
  volume in that specific session*, controlling for a structure we already
  know exists rather than letting it contaminate the signal.
- **Position in own cycle.** Drawdown from trailing 30d/90d peak. Orthogonal
  to momentum: momentum measures recent *change*, this measures *level
  relative to history*.
- **Volume share rotation.** This coin's share of total universe volume, and
  the **change** in that share. Attention rotating between coins is a real
  cross-sectional phenomenon, measured directly.
- **Coin age.** Days since listing. Young coins behave differently; SUI is not
  BTC partly for this reason alone.
- **Beta momentum.** β is *removed* from the target, but the **change** in β is
  a legitimate feature — a coin becoming more systematic is informative.
- **Beta instability.** The within-window variance of rolling β. A coin whose
  market relationship is unstable has a less reliable residual — which also
  flags that its label is noisier.
- **Idio-vol momentum.** Low realised vol + rising OI = a coiled spring.
- **Rank persistence.** How long has this coin sat at the top or bottom of the
  cross-section?
- **BTC catch-up.** Given BTC's move, where *should* this coin be, and where is
  it? The previous project maintained a whole `beta_catchup` artifact with
  1h–12h gaps. Its conclusions are retracted; the *idea* is not, and it is
  cheap to rebuild cleanly.

---

## Seam 9 — Relational / network

Treats the 20 coins as a system rather than 20 independent rows. The target
removes **one** factor (the market). Everything between that single factor and
pure idiosyncrasy — sectors, clusters, lead-lag chains — is unexploited.

### 9.1 Sector-relative residual momentum — the lead idea

**This is not a hypothesis. Step 3 already measured it.**

The PC2 loadings from the factor-structure test:

```
LONG    most positive: XRP +0.45, XLM +0.38, DOGE +0.29, TRX +0.26
        most negative: NEAR -0.32, AAVE -0.30, SOL -0.29, UNI -0.26
ALL20   most positive: XRP +0.57, XLM +0.57, ADA +0.19
        most negative: ETH -0.21, SUI -0.20, AAVE -0.19, BTC -0.17
```

**XRP and XLM load together at +0.38 to +0.57 in both independent samples** —
the two payment/remittance coins, shared narrative and founder lineage,
co-moving beyond the market factor. Against them sits the newer L1/DeFi
cluster.

That is a sector factor, measured in this project's own residuals. PC2 came in
at 4.6–5.4%, correctly *below* the 15% two-factor threshold, so the book is
**not** neutralised to it. But "too small to hedge" and "too small to trade"
are different statements. Sector-relative features are the natural way to
harvest what single-factor neutralisation leaves behind.

#### Construction

```
sector_rel_i,t  =  ε_i,t  −  mean( ε_j,t  :  j ∈ sector(i),  j ≠ i )
```

evaluated over trailing windows on the geometric ladder (8h / 24h / 72h /
168h), with `ε` the idio-vol-normalised residual return — i.e. **the same
space as the target**.

**The `j ≠ i` exclusion is not optional.** With a sector of three, including
self makes the peer mean one-third yourself, mechanically damping the feature
by that fraction — and damping it *differently for every sector size*, which
turns sector size into a spurious cross-sectional signal. Leave-one-out
throughout.

#### The sector-beta refinement

The construction above silently assumes **sector beta = 1** — that a coin
moves one-for-one with its peers. That is not how the market leg is handled:
there, β is estimated and the residual formed properly. The consistent version
is:

```
sector_rel_i,t  =  ε_i,t  −  β_sector,i · mean( ε_j,t : j ∈ sector(i), j ≠ i )
```

with `β_sector,i` the coin's beta to its own sector, estimated on 1h residuals.

**The trade-off is familiar but the answer may differ.** Estimating another
beta costs precision, and with 2–6 peers the sector mean is already noisy.
Step 5 hit this same bias-variance question — but Step 9 resolved it for a
**label**, where bias becomes systematic exposure and unbiasedness therefore
dominates. This is a **feature**, not a label. Nothing here has to be
unbiased; it only has to predict. A shrunk or simple-subtraction version may
genuinely win.

Build both, compare as a one-variable experiment, and do not assume Step 9's
answer transfers.

#### Sector map — hand-assigned, and deliberately so

With 20 coins there are **190 pairwise correlations** to estimate. A hand
assignment is a *prior*; an estimated clustering is a noisy estimate of 190
parameters. In samples this small, the prior wins. The groupings below are
economically obvious rather than fitted:

| sector | members | n |
|---|---|---|
| Majors | BTC, ETH | 2 |
| Smart-contract L1 | SOL, AVAX, NEAR, DOT, ADA, SUI, HBAR | 7 |
| PoW / legacy | LTC, BCH, ZEC | 3 |
| Payments | XRP, XLM | 2 |
| DeFi / oracle | UNI, AAVE, LINK | 3 |
| Other | BNB, DOGE, TRX | 3 |

#### Expected sign — stated before testing

Step 7L measured **reversal** at this horizon: a parameter-free reversal probe
scored IC +0.0133 (t = 2.91 after block bootstrap) at 8h with a 60min lag.

So the expectation is **reversal at the short end** — a coin that
underperformed its sector over 8–24h should outperform next — with a possible
**crossover to momentum at 72–168h**. Where that crossover sits, if it exists,
is the interesting result. Recorded now so it cannot be retrofitted.

#### How it could fail

1. **Small sectors give weak or degenerate features.** Majors and Payments
   have two members each, so the "sector-relative" return collapses to a pure
   pair spread. That is a legitimate feature but a different one, and it should
   be understood as such rather than pooled silently with the 7-member L1
   sector.
2. **Sector assignment is a researcher degree of freedom.** TRX is genuinely
   ambiguous — a smart-contract L1 by technology, but PC2 places it with the
   payments/legacy side. **The map above must be frozen before any IC is
   computed.** Re-assigning coins after seeing results turns the sector map
   into a tuning knob with 20 dials.
3. **Few peers means a noisy peer mean.** A 3-coin sector estimates its mean
   from 2 observations. Sector-relative return for those coins is barely more
   than a pairwise spread.
4. **Sector effects may already be absorbed.** PC2 is only ~5% of variance;
   the market factor already removed 55–63%. There may not be much left.
5. **Survivorship bias in the sector composition.** The universe is today's
   survivors, so the sectors are survivor sectors — dead L1s are absent, and
   the surviving ones are not a random sample of what the sector was in 2021.

#### A continuous alternative, avoiding the judgment calls

Rather than discrete buckets, use each coin's **PC2 loading as a continuous
sector coordinate**, and define the feature as the residual minus a
loading-weighted average of peers. This removes the hand-assignment degree of
freedom entirely, at the cost of requiring PC2 to be estimated
**point-in-time** on a long trailing window — the full-sample PC2 quoted above
is lookahead and cannot be used directly.

Worth building both and comparing, as a one-variable experiment.

### 9.2 Sector correlation and cohesion

§9.1 measures the **level difference** — how far a coin has moved from its
peers. It says nothing about how **tightly coupled** it currently is to them.
Those are different quantities, and the second conditions the first.

**Why they are natural partners.** Sector-relative momentum implicitly assumes
the sector relationship is meaningful. If a coin normally tracks its peers at
ρ = 0.8, a 2% deviation is a real signal. If it tracks them at ρ = 0.2, that
same 2% is mostly noise. **Correlation tells you how much to trust the level
difference.**

| feature | construction | story |
|---|---|---|
| **Coin-to-sector correlation** | `ρ(ε_i, mean(ε_peers))` on 1h residuals, leave-one-out | is this coin moving with the pack or independently? |
| **Decoupling** | current ρ vs that coin's *own* trailing average ρ | a coin that normally tracks at 0.75 and is suddenly at 0.30 is doing *something* — whether that persists (news) or reverts (noise) is the empirical question |
| **Sector cohesion** | average pairwise correlation *within* the sector | is the sector trading as a bloc or fragmenting? |
| **Sector dispersion** | cross-sectional spread of residuals inside the sector | high dispersion = intra-sector opportunity |
| **Belonging** | ρ to own sector vs ρ to other sectors | how well does the hand-assigned map actually fit? |

#### Belonging is a diagnostic, not a re-fitting tool

If TRX correlates more with the L1 sector than with its assigned "Other"
bucket, that is worth knowing and worth reporting. It is **not** licence to
move TRX. Re-assigning coins after seeing results is precisely the 20-dial
tuning knob §9.1 warns against — the map is frozen before any IC is computed,
and a poor fit is recorded as a limitation rather than repaired mid-flight.

#### A structural point: sector features are a middle tier

§12 establishes that market-wide context features have **zero** ranking power,
because they take the same value for every coin in a cross-section.

**Sector-level features are not like that.** Sector cohesion is shared *within*
a sector but **differs across sectors** — with six sectors, coins in a single
bar receive six distinct values. That is coarse discrimination, but it is
genuine within-bar discrimination, and it is not available to any market-wide
aggregate.

So there are three tiers, not two:

| tier | example | ranking power |
|---|---|---|
| Per-coin | funding, path efficiency | full |
| **Per-sector** | **sector cohesion, sector dispersion** | **partial — 6 distinct values per bar** |
| Bar-level | market dispersion, avg pairwise correlation | **none** (interactions only, §12) |

The middle tier is the one worth noticing: it buys real ranking power at a
fraction of the estimation cost of a per-coin network measure, because a
sector mean over 3–7 coins is far better determined than a 20×20 matrix.

### 9.2b Correlation to BTC — and a measured trap

Rolling correlation of a coin to **BTC specifically** is distinct from
everything else in the pool: beta (Seam 8) is measured against the *index*,
and §9.3's BTC/ETH features are *lagged*. Nothing tracks contemporaneous
coupling to BTC.

It is worth having, because **the book is neutral to `r_m`, not to BTC.**
`corr(r_m, r_BTC) = 0.948`, so ~10% of BTC's variance is orthogonal to the
index and residuals could carry it.

**But note first: plain rolling `corr(coin, BTC)` on raw returns is already
covered.** It is 95% redundant with `σ_ε/σ_total` for ranking purposes — see
§8.2. Only the *residual* version discussed here is additional, and it carries
the trap below.

#### The trap — measured 2026-08-07, not hypothesised

The obvious construction, `corr(εᵢ, ε_BTC)` using the standard index, is
**almost entirely a mechanical artifact**:

| construction | mean corr across 19 coins | sign split |
|---|---|---|
| Index **includes** BTC | **−0.2585** (range −0.39 to −0.12) | 0 positive / 19 negative |
| Index **excludes** BTC | **+0.0143** (range −0.12 to +0.24) | 9 positive / 10 negative |

Step 6 established `Σ wᵢ εᵢ = 0`. BTC carries 51% weight, so `ε_BTC` is
mechanically the negative weighted sum of every other residual — and every
coin therefore correlates negatively with it *by construction*. Removing BTC
from the index breaks the constraint and the effect collapses, moving the mean
by **+0.273**.

**A naive version of this feature would have looked spectacular** — every coin
loaded, `|t|` up to 76 — and measured nothing but the constraint. **Any
correlation-to-BTC feature must be computed against a leave-one-out index.**

The same caution applies to §9.2's coin-to-sector correlation for any
high-weight coin, and to network centrality in §9.4.

#### What survives is the sector factor, again

The structure remaining after the artifact is removed is small but ordered:
**ETH +0.237, LTC +0.139, BCH +0.125** (majors and PoW forks) against
**XRP −0.122, SUI −0.077, XLM −0.054** (payments, newest L1).

That is the same split PC2 found in Step 3, reached by an entirely different
route. Two independent measurements agreeing is worth more than either alone —
and it **independently validates the single-factor decision**: the genuine
residual structure is small (mean +0.014), consistent with PC2's measured
4.6–5.4%.

### 9.3 Lead–lag propagation

Does one coin's residual predict another's next bar? In crypto, information
plausibly flows BTC/ETH → large alts → small alts.

**Do NOT estimate a 20×20 matrix.** That is 400 directed pairs — more
parameters than the data supports, and precisely the construction that yields
a beautiful backtest and nothing else.

**Collapse it to the hypothesis instead:** use *BTC's and ETH's last-bar
residuals* as features for every other coin. **Two features rather than 400.**
It tests the propagation story directly at near-zero multiple-testing cost. If
two features cannot find signal, four hundred will find only overfitting.

### 9.4 Remaining ideas in this seam

- **Peer-spread z-score** on *a priori* pairs — XRP/XLM especially, since PC2
  says they genuinely co-move. Also LTC/BCH, UNI/AAVE.
- **Network centrality** — eigenvector centrality of the residual correlation
  matrix. Central coins are quasi-systemic; peripheral ones carry more genuine
  idiosyncratic risk. Changes in centrality flag regime shifts.
- **Dispersion contribution** — how much of today's cross-sectional spread
  this coin accounts for. A coin that *is* the dispersion is the outlier.
- **Rank dynamics** — velocity through the cross-sectional ranking, and time
  spent in the current tercile.

### 9.5 The estimation problem, quantified

Every idea here rests on structure estimated across a 20-name cross-section:

| estimation frequency | obs in 30d | SE of each pairwise correlation |
|---|---|---|
| 8h returns | 90 | **0.105** |
| 1h returns | 720 | **0.037** |

At 8h, distinguishing a true correlation of 0.3 from 0.4 is hopeless — the
noise exceeds the differences one would act on. **Estimate network structure
on 1h residuals, never 8h.** Step 4 established there is no Epps effect in this
universe, so 1h is clean for this purpose.

**Three mitigations, in order of how much weight they deserve:**

1. **Prefer priors to estimates.** Hand-assigned sectors over estimated
   clusters. The single most important call in this seam.
2. **Collapse matrices to hypotheses.** BTC/ETH lead–lag as 2 features, not a
   400-cell matrix.
3. **Separate timescales.** Network *structure* is slow-moving — estimate it on
   90–180d. The *signal* is fast — compute it on the short ladder. Estimating
   both on the same 30d window discards the stability structure actually has.

### 9.6 A caution on the record

**This seam is the most intellectually seductive and the most likely to be
noise.** Network methods manufacture impressive-looking structure out of pure
randomness: a correlation matrix of 20 independent series still has a largest
eigenvalue, still has clusters, still has a most-central node. None of it means
anything.

**Build the prior-based version first. Reach for estimated structure only if
the simple version works.**

---

## 12. Bar-level context features work ONLY through interactions

A caveat that changes how several ideas above should be used.

Market-wide dispersion, average pairwise correlation, market funding level,
market OI change — these are **identical for every coin in a cross-section**.
Under a `lambdarank` objective they therefore **cannot affect the within-bar
ordering at all**. On their own they have exactly zero ranking power.

They remain worth having, but only as **interaction terms**: a tree model can
learn *"when correlation is high, weight reversal less."* Step 6 gives a
concrete reason to care — idiosyncratic share fell from 0.708 to **0.567** in
2022. Cross-sectional opportunity genuinely shrinks in stress, and a model
that knows this can behave differently.

If they are included, include them knowing they only work through
interactions. Otherwise they are wasted slots against the multiple-testing
budget.

### But there are three tiers, not two

§9.2 identifies a middle tier this caveat would otherwise obscure:

| tier | example | ranking power |
|---|---|---|
| Per-coin | funding, path efficiency, variance ratio | full |
| **Per-sector** | **sector cohesion, sector dispersion** | **partial — 6 distinct values per bar** |
| Bar-level | market dispersion, avg pairwise correlation | **none** — interactions only |

Sector-level aggregates are shared *within* a sector but **differ across
sectors**, so with six sectors a single cross-section receives six distinct
values. Coarse, but genuine within-bar discrimination — and unavailable to any
market-wide aggregate.

That middle tier is the efficient one: it buys real ranking power at a
fraction of the estimation cost of a per-coin network measure, because a
sector mean over 3–7 coins is far better determined than a 20×20 matrix
(§9.5).

---

## 13. Where I would bet

From the first pass:

1. **Retail-attention divergence** (Seam 4) — novel, free, clean story,
   directly cross-sectional.
2. **Spot–perp lead–lag** (Seam 5) — genuinely unavailable in other asset
   classes.
3. **Path efficiency + volume concentration** (Seam 3) — capture *how* a move
   happened rather than only how big it was.

Added on the second pass:

4. **Premium mean-reversion speed** (Seam 2) — novel, real economics, measures
   arbitrage capacity per coin.
5. **Cumulative funding burden** (Seam 1) — horizon-matched, captures
   accumulated stretch rather than instantaneous level.
6. **Zero-return minutes** (Seam 6) — nearly free, literature-grounded, and a
   single query establishes whether it is alive in this universe.

Added on the relational pass:

7. **Sector-relative residual momentum** (Seam 9.1) — the only idea in this
   pool supported by evidence *already measured in this project*: PC2's
   loadings are a sector factor, with XRP and XLM at +0.38 to +0.57 in two
   independent samples. Robust by construction (hand-assigned priors, not
   estimated clusters), and industry-relative momentum is the most replicated
   cross-sectional feature in quant finance.
8. **BTC/ETH lead–lag as exactly 2 features** (Seam 9.3) — tests a real
   crypto-specific propagation story at near-zero multiple-testing cost.

Added on the third pass:

9. **Variance ratio** (Seam 8.1) — conditions the benchmark signal itself.
   Reversal scores IC 0.0133; VR says *where* it should work. Cheap,
   well-grounded, and nothing else in the pool separates a mean-reverting coin
   from a trending one.
10. **Multi-horizon agreement** (§14.6) — the only idea here that *reduces*
    the feature count rather than adding to it, while encoding conviction
    across the lookback ladder.

Added on the fourth pass:

11. **Coin-to-sector correlation** (Seam 9.2) — the natural partner to §9.1.
    Sector-relative momentum assumes the sector relationship is meaningful;
    correlation says how much to trust it. A 2% deviation from peers means
    something very different at ρ = 0.8 than at ρ = 0.2.
12. **Sector cohesion / dispersion** (Seam 9.2) — the per-sector middle tier
    (§12). Real within-bar ranking power at a fraction of the estimation cost
    of a per-coin network measure.

Added on the fifth pass:

13. **Crowd-winning: funding surprise × return surprise** (Seam 1.1) — a
    single continuous axis for "is the crowded side winning or losing",
    horizon-matched, with a measured justification for every construction
    choice and an expected sign recorded in advance.
14. **Horizon-matched BTC correlation, `corr₈ₕ(90d)`** (§8.2) — persistent
    (+0.218 / +0.411) and only 70% redundant with idio share, versus 95% for
    the 1h version.

All fourteen come from data already on disk. No backfill required.

**One candidate was tested and rejected before reaching this list:** the
correlation *term structure* (§8.2), which looked almost purely additive at
+0.295 redundancy and turned out to be noise (persistence +0.04). It is
recorded rather than deleted, because the way it nearly passed is more
instructive than the fact that it failed.

---

## 14. What must happen before any of this is tested

> **The screening funnel is specified in `docs/FEATURE_EVALUATION.md`** —
> six target-free stages with pre-registered thresholds (validity and leakage,
> ranking power, estimation quality, persistence, redundancy, implied cost),
> the freeze point, and what remains open on the target-touching side. The
> rules below are the principles it implements.


Recorded now, while nothing has been computed and there is nothing to
rationalise:

1. **Freeze the list first.** A separate pre-registered document naming every
   feature, its construction, and its **expected sign**, written before the
   first IC is computed.
2. **Group by mechanism, not by column.** Multiple-testing correction should
   be over *families* (carry, positioning, microstructure, path, attention,
   network), not over the dozens of variants inside each. Otherwise the
   correction punishes thoroughness rather than fishing.
3. **Budget 20–40 features, not 110.** At 2σ, testing 100 features yields ~5
   false positives by construction. The previous project tested 230 and the
   full set lost to a subset in 10/10 seeds — not because the features were
   bad, but because that many draws against this much noise is unwinnable.
4. **Benchmark against reversal (IC 0.0133), never against zero.**
5. **Screen every candidate for beta-proxying** via the regime split, per
   `BETA_NEUTRAL_DESIGN.md` D3 — features are screened, not residualised
   wholesale.

   **And screen for redundancy AND persistence, in that order.** Redundancy
   against already-computed quantities (§8.2 found a candidate 95% duplicated
   by an existing column). Then persistence: does the ordering a feature
   produces survive to a non-overlapping period? **Low redundancy has two
   explanations — new information or noise — and noise correlates with
   nothing**, so a redundancy check alone cannot distinguish them. §8.2 records
   a candidate that looked almost purely additive (+0.295 redundancy) and was
   noise (persistence +0.04).
6. **Lookbacks on a geometric ladder** (8h / 24h / 72h / 168h), anchored to the
   8h horizon. Adjacent lookbacks correlate ~0.95 and add columns without
   adding information while paying full multiple-testing cost. Prefer
   *ratios* (short ÷ long) to carrying both levels.

   **Multi-horizon agreement — the better answer to the ladder.** There is a
   real tension here: several horizons carry information, but multiplying every
   feature by four lookbacks burns the entire budget. Agreement compresses
   them into one column:

   ```
   agreement = sign(sig_8h) + sign(sig_24h) + sign(sig_72h) + sign(sig_168h)
   ```

   Range −4 to +4, **one feature slot instead of four**, and it directly
   encodes *conviction*: all four horizons pointing the same way is a
   materially different state from two-against-two. That distinction is
   invisible to any single-horizon feature and expensive to recover from four
   separate ones.

   *What it gives up:* magnitude. A feature at +4 with tiny moves at every
   horizon looks identical to +4 with large ones. And if only one horizon
   actually carries signal, agreement **dilutes** it by averaging in three
   noisy votes — so it should be compared against the single best horizon
   rather than assumed superior. A signed-magnitude variant (mean of the
   standardised signals) is the natural middle ground and costs the same one
   slot.

7. **Encode known interactions explicitly rather than hoping a tree finds
   them.** Trees *can* learn interactions, but they need data to do it, and
   this project is sample-constrained at 4,558 cross-sections. Where an
   interaction is already established — reversal being weaker on high volume
   (Campbell–Grossman–Wang), or stronger where variance ratio is low —
   constructing it directly costs one slot instead of asking the model to
   discover it.

---

## 15. Open questions

1. Backfill liquidations? It fits the horizon and the mechanism is clean
   (forced selling is price-insensitive, so it reverts), but the data is not
   yet downloaded.
2. Backfill aggTrades? Would sharpen Seam 6 considerably, at a large storage
   cost. Roll and Kyle proxies from klines may capture most of it for free —
   worth testing the cheap version first.
3. Start recording **L2 order-book depth** now? It cannot be backfilled, so
   the only way to ever have history is to begin collecting. Irrelevant to
   this research cycle; decisive for the next one.

4. **Realized semi-beta — this is target work, not feature work.** Estimate β
   separately on up-market and down-market 1h returns and compare `β⁻` to
   `β⁺`.

   As a *feature* it is unremarkable: a coin with high downside beta is
   asymmetrically risky. As a **diagnostic on the target** it is more
   interesting, because it probes a gap Gate 9B did not cover.

   Gate 9B tested whether `corr(target, β)` **flips sign** between up and down
   markets — a *linear* leakage test — and it passed cleanly (gap +0.0008).
   It did **not** test whether the residual carries *asymmetric* loading. If
   `β⁻ ≠ β⁺` materially, then `r − β·r_m` leaves a directional component in
   the label that a symmetric linear test cannot see: the residual would be
   systematically negative in down markets and positive in up ones for
   high-`β⁻` coins, while the *correlation* with β stays flat.

   Cheap to measure with `mft/rolling.py` (it already returns the centered
   moments needed). Worth doing before the feature programme rather than
   after, because a finding here changes the target rather than the feature
   list. Recorded in `docs/TARGET_DESIGN.md` §9 as an open item.
