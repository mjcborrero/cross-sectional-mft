# Frozen feature list — pre-registration

**Status: FROZEN 2026-08-07. AMENDED 2026-08-07** (same day, before any target
contact — see §9). Nothing below has been computed. No IC has been measured.

Companions: `docs/FEATURE_EXPLORATION.md` (the pool this was drawn from),
`docs/FEATURE_EVALUATION.md` (the screening funnel), `docs/TARGET_DESIGN.md`
(the label).

---

## 1. What freezing means

**This list may not grow.** Features may be *removed* by the target-free
stages of `FEATURE_EVALUATION.md` §2–7 — those stages never see the target, so
removal by them costs nothing and biases nothing. No feature may be *added*
after this date without the addition being recorded as an amendment (§9), and
amendments are counted as additional tests.

**Every entry states its expected sign in advance.** Where the prior is weak
that is marked, because "I expected either sign" written afterwards is not a
prediction.

**All features are computed on PERPETUAL futures data** (`perp_klines`,
`funding`, `metrics`, `premium_index`) unless a construction explicitly names
spot. Family G is the only place spot appears, and there it is always a
perp-vs-spot *contrast*, never spot alone — the book trades perps, so a
spot-only feature would predict an instrument that is not held.

**Sign convention.** IC is measured against the target — a standardised
idiosyncratic forward return. *Positive IC* means higher feature value
predicts higher forward idiosyncratic return. Constructions are written in
their natural form; the expected sign is stated, not engineered to be positive.

**Everything here is computable from data already on disk.** Nothing requires
the liquidation or aggTrades backfills.

---

## 2. Structure — two parts, treated differently

**Part 1 (§3): designed features, 67.** Every idea brainstormed in
`FEATURE_EXPLORATION.md`, none dropped. Hypothesis-driven, each with a stated
expected sign. These are predictions.

**Part 2 (§4): the inherited library, ~260 columns.** The old project's base
features, **clock-audited** (§4.1) — 65 of 101 admitted, 36 excluded because
the dollar bars were load-bearing — times the per-coin relative transforms. These are **not** hypotheses — no individual expected sign is
claimed for them. They are a screen population, and treating them as
predictions would be dishonest since nobody wrote down what they should do.

The distinction matters for interpretation. A designed feature that fails is
evidence against its mechanism. An inherited feature that fails is evidence of
nothing in particular.

---

## 3. Part 1 — designed features

### Family A — Carry (5)

| # | feature | construction | expected | conf |
|---|---|---|---|---|
| A1 | `funding_z` | `(funding − rolling_mean₃₀d) / rolling_std₃₀d` | **−** crowded longs revert | high |
| A2 | `funding_cum_7d` | Σ funding paid, trailing 7d | **−** accumulated stretch | med |
| A4 | `funding_sign_persist` | signed count of consecutive same-sign settlements | **−** entrenched crowding | med |
| A5 | `crowd_winning` | `z_funding × z_return` (§1.1 of the pool) | **−** crowd winning is fragile | med |

Demeaning in A1 is not cosmetic: funding's `|rolling mean| / rolling std` is
**0.904**, so without it the feature ranks coins on a permanent characteristic.

### Family B — Positioning (5)

| # | feature | construction | expected | conf |
|---|---|---|---|---|
| B1 | `oi_velocity_24h` | `ΔOI / OI` over 24h | **−** position building precedes unwind | low |
| B2 | `oi_price_corr` | rolling `corr(ΔOI, Δprice)`, 30d | **−** new longs on up-moves revert | med |

| B6 | `oi_per_trade` | `ΔOI ÷ trade_count` | **−** many small builders = retail | low |
| B7 | `turnover` | `volume ÷ OI` | **+** fast churn, weak hands | low |

> **B3, B4, B5 REMOVED at Stage 0, 2026-08-10 — data unavailable, not
> rejected.** Binance stops publishing `sum_toptrader_long_short_ratio` and
> `sum_taker_long_short_vol_ratio` for essentially all of 2022 (~99% null,
> Q1–Q4). That is a hole *inside* the window where the dataset otherwise
> exists, so no coverage-denominator correction can excuse it: measured
> coverage was **68.6%** against a 95% bar, having risen from 51.1% once the
> denominator was fixed — which is what confirms the cause is the 2022 gap
> rather than a build defect.
>
> **The mechanism was never tested.** Nothing is claimed about whether
> smart-money positioning predicts. Worth restoring if Binance ever backfills
> those fields.
>
> Also worth recording: the gap covers 2022 exactly, the stress year where
> Step 6 measured idiosyncratic share collapsing to 0.567. These features
> would have been blind precisely where positioning matters most.

Family B carries the weakest priors in the list. That is recorded rather than
hidden — five of seven are marked low confidence, and a null result here is
informative rather than disappointing.

### Family C — Liquidity and microstructure (5)

| # | feature | construction | expected | conf |
|---|---|---|---|---|
| C1 | `zero_return_minutes` | count of the 480 1m bars with no price change | **+** illiquidity premium | low |
| C3 | `roll_spread` | `2√(−cov(r_t, r_{t−1}))` on 1m returns | **+** wider spread, more reversion | low |
| C4 | `kyle_lambda` | within-bar regression of 1m returns on signed volume | **+** impact-sensitive reverts | low |
| C6 | `vpin_perp_z` | perp signed VPIN, z-scored 30d *(amendment)* | **−** toxic flow precedes reversal | med |
| C5 | `premium_reversion_speed` | AR(1) coefficient on the 1m premium index | **−** slow reversion = low arb capacity, dislocation persists | med |

| C8 | `corwin_schultz` | high-low spread estimator, adjacent periods | **+** cross-check on C3 | low |

> **Built 2026-08-10 — all 8 passed Stage 0/1 and the truncation test.**
> Two notes recorded at build time:
>
> **`roll_spread` is floored at zero in 37.8% of observations.** Roll's
> estimator is undefined when serial covariance is positive, which is common
> at 1-minute resolution in trending markets; the standard treatment floors it
> at zero rather than discarding the observation. It still passes the
> distinct-values gate (median 14 of ~19) but carries a large tie group at
> zero, so its effective ranking resolution is materially below the other C
> features. C8 `corwin_schultz` was designed as the cross-check on exactly
> this, and Stage 4 redundancy will decide whether both survive.
>
> **`zero_return_minutes` is alive**, contrary to the §8 risk note predicting
> it might be identically zero in a universe this liquid. Median 16 distinct
> values per bar — coins genuinely differ in how often a minute passes without
> a price change. The risk was real; the outcome was not.

C1–C4 are expected to be weak *standalone* and to earn their keep as
conditioners. They are listed as standalone features anyway so that claim is
testable rather than assumed.

### Family D — Path and attention (5)

| # | feature | construction | expected | conf |
|---|---|---|---|---|
| D1 | `retail_attention_div` | `log(trade_count / mean) − log(volume / mean)` | **−** retail-driven reverts | med |
| D3 | `volume_concentration` | Herfindahl of the 480 per-minute volumes | **−** burst-driven reverts | med |
| D4 | `close_vs_vwap` | `(close − intra-bar VWAP) / VWAP` | **+** buyers held into the close | med |
| D7 | `move_timing` | share of the 8h return realised in the first hour vs last | **+** early-and-held beats late | med |
| D8 | `realized_skew` | 3rd moment of the 480 1m returns | **+** crash-shaped moves bounce | low |
| D9 | `realized_kurt` | 4th moment of the 1m returns | **−** fat-tailed bars revert | low |
| D10 | `parkinson_ratio` | high-low range vol ÷ close-to-close vol | **−** gappy movement reverts | low |
| D11 | `volume_return_coupling` | within-bar `corr(\|1m return\|, 1m volume)` | **−** volume-driven reverts | low |

### Family E — Return and regime (6)

| # | feature | construction | expected | conf |
|---|---|---|---|---|
| E1 | `resid_reversal_8h` | `−(residual return 8h) / σ_ε` — **the benchmark** | **+** measured at IC +0.0133 | high |
| E3 | `variance_ratio_72h` | `Var(72h) / (9 × Var(8h))` on 1h residuals, 90d | **−** trending coins revert less | med |
| E4 | `reversal_x_vr` | E1 × (1 − E3), explicit interaction | **+** reverse harder where reversion is real | med |

| E7 | `beta_momentum` | change in β over 30d | **?** becoming more systematic | low |
| E8 | `beta_instability` | within-window variance of rolling β | **−** unstable β = noisier residual | low |
| E9 | `idio_vol_momentum` | change in `σ_ε` over 30d | **−** vol expansion reverts | low |
| E10 | `drawdown_from_peak` | distance below trailing 30d peak | **+** position in own cycle | med |
| E13 | `volume_share_rotation` | change in this coin's share of universe volume | **−** attention rotation reverts | med |
| E14 | `session_rel_volume` | volume vs this coin's own norm *for that UTC session* | **−** | low |

> **Built 2026-08-10 — 13 features, all gates passed, truncation 0.00e+00.**
> Two changes forced by Stage 1, both recorded:
>
> **E2 switched from sign-count to signed magnitude.** A sum of four signs can
> take only five values, which tied ~4 of 19 coins per bar and failed the
> resolution gate (median 5 distinct, bar 10). The signed-magnitude variant was
> named in advance in `FEATURE_EXPLORATION.md` §14.6 as *"the natural middle
> ground and costs the same one slot"* — so the gate found the coarseness and
> the replacement was already pre-specified, not invented to pass.
>
> **E12 `rank_persistence` REMOVED — insufficient ranking resolution.** Also
> discrete by construction (median 6 distinct values). Unlike E2 it had no
> pre-specified continuous alternative, and inventing one after seeing the
> failure would be retrofitting. Recorded as removed at Stage 0, which costs
> nothing statistically.

E4 and E5 encode interactions explicitly rather than hoping a tree finds them,
per `FEATURE_EVALUATION.md` and the sample constraint. E14 controls for the
Asia/Europe/US structure the 8h grid already encodes — raw session is excluded
(§7) because it is bar-level.

### Family F — Relational (5)

| # | feature | construction | expected | conf |
|---|---|---|---|---|
| F1 | `sector_rel_reversal_8h` | `−(ε_i − mean(ε_peers, j≠i)) / σ_ε`, hand-assigned sectors | **+** reversal within sector | med |
| F2 | `sector_rel_agreement` | **signed magnitude** across 8/24/72/168h *(same revision as E2)* | **+** | med |
| F3 | `coin_sector_corr` | `ρ(ε_i, mean(ε_peers))`, 90d, leave-one-out | **+** tighter coupling, more meaningful deviation | low |
| F4 | `btc_resid_lag` | BTC's previous-bar residual, applied to every other coin | **+** propagation BTC→alts | med |
| F5 | `corr_btc_8h_90d` | rolling `corr(r_i, r_BTC)`, 8h returns, 90d window | **−** | low |

| F6 | `sector_cohesion` | mean pairwise ρ within the sector, 90d | **?** per-sector tier (§4.2) | low |
| F7 | `sector_dispersion` | cross-sectional spread of residuals inside the sector | **+** intra-sector opportunity | low |
| F8 | `sector_belonging` | ρ to own sector − ρ to other sectors | **?** also a map diagnostic | low |
| F9 | `eth_resid_lag` | ETH's previous-bar residual | **+** propagation | med |
| F10 | `sector_decoupling` | current ρ to sector vs own trailing ρ | **?** news or noise — the open question | low |
| F11 | `dispersion_contribution` | share of the bar's dispersion from this coin | **−** outliers revert | low |

> **Built 2026-08-10 — 11 features, all gates passed, truncation 0.00e+00.**
>
> This family spans **all three ranking tiers** of `FEATURE_EXPLORATION.md`
> §12, and the build declares which is which rather than letting Stage 1
> discover it:
>
> | tier | features | treatment |
> |---|---|---|
> | per-coin | F1 F2 F3 F5 F8 F10 F11 | full gate |
> | **per-sector** | **F6 F7** | 6 sectors → median 6 distinct values; gate lowered to ≥4 **for declared sector columns only** |
> | bar-level | F4 F9 | exempt from ranking gates; interaction use only |
>
> The per-sector tier needed a gate change, and it is worth being explicit that
> this is not a threshold loosened to make features pass. §12 established the
> three tiers *before* any feature was built, and argued the middle tier is the
> efficient one — real within-bar discrimination at a fraction of the
> estimation cost of a per-coin network measure. Holding it to the per-coin
> threshold would have deleted a tier the design deliberately keeps. The tier
> is **declared by the family in code**, so it cannot be claimed after seeing a
> failure.
>
> F2 uses signed magnitude for the same reason as E2 — applied proactively
> here rather than walking into a known failure.

### Family G — Spot vs perp, basis and premium (11)  *(added by amendment)*

| # | feature | construction | expected | conf |
|---|---|---|---|---|
| G1 | `basis_z` | perp−spot basis, z-scored on 30d | **−** rich perp reverts | med |
| G3 | `perp_spot_vol_ratio` | perp quote_volume ÷ spot quote_volume, z-scored | **−** leverage froth | med |
| G4 | `aggressor_divergence` | taker-buy share in perp − same in spot | **−** leveraged longs vs spot sellers is fragile | med |
| G5 | `perp_spot_return_gap` | perp return − spot return over the bar | **−** mean-reverting dislocation | med |
| G6 | `perp_spot_leadlag` | cross-correlation of 1m perp vs spot at ±k, 7d | **−** perp-led moves less durable | med |
| G7 | `perp_spot_tradecount_ratio` | perp ÷ spot `trade_count` | **−** participant-mix skew | low |
| G8 | `premium_vol` | volatility of the 1m premium index | **−** unstable perp-spot link | med |
| G9 | `premium_time_above_zero` | fraction of the 480 minutes with premium > 0 | **−** sustained richness | med |
| G10 | `premium_range` | premium high−low ÷ \|mean premium\| | **−** dislocation amplitude | low |
| G11 | `term_basis_slope` | quarterly−perp basis slope *(BTC/ETH only — see §7)* | **−** | low |

Added because Seam 5 was listed as bet #2 in `FEATURE_EXPLORATION.md` §13 and
was omitted from the original freeze by oversight, not judgment.

**The sector map is frozen with the list:**

| sector | members |
|---|---|
| Majors | BTC, ETH |
| Smart-contract L1 | SOL, AVAX, NEAR, DOT, ADA, SUI, HBAR |
| PoW / legacy | LTC, BCH, ZEC |
| Payments | XRP, XLM |
| DeFi / oracle | UNI, AAVE, LINK |
| Other | BNB, DOGE, TRX |

TRX is the known ambiguity — an L1 by technology, but PC2 places it with the
payments/legacy side. **It stays in "Other."** Re-assigning after seeing
results would turn the map into a tuning knob with 20 dials.

F5 must use a **leave-one-out index** wherever a residual is involved: the
naive version scored |t| up to 76 and was pure artifact of `Σwε = 0`.

---

## 4. Part 2 — the inherited library

The old project's feature code survives in `mft/features/`. Its *conclusions*
about which features worked were retracted; its *implementations* are
arithmetic and are reusable.

### 4.1 Clock audit — the eligibility test

**Rule: a feature computed on dollar bars is admitted only if its construction
survives the move to the 8h wall-clock grid.** If the dollar bars are load-
bearing rather than incidental, the feature is excluded — not ported, not
approximated.

The evidence is in the modules' own docstrings.

#### EXCLUDED — the dollar clock is the construction (36 features)

| module | n | why it cannot port |
|---|---|---|
| `vpin.py` | **24** | *"The dollar-volume bars **ARE** the VPIN volume buckets: every bar holds the same ~200M USDT of perp quote volume, so per-bar taker-buy share gives the bucket imbalance directly."* Remove equal-volume bars and the bucket imbalance has no definition. |
| `spot_flow.py` | **12** | *"Spot aggregated into the same **BTC-clock windows** as the perp bars."* Built on a clock this project abandoned. |

**The mechanisms are not lost — only these implementations.** Order-flow
toxicity survives as **C6 `vpin_perp_z`**, re-bucketed by volume *inside* each
8h window, and spot-side flow survives as **G4 `aggressor_divergence`**. Both
are new constructions written for this clock, not ports.

#### ADMITTED — wall-clock native or clock-agnostic (65 features)

| module | n | evidence |
|---|---|---|
| `open_interest.py` | 13 | 5-minute `metrics` on its own clock |
| `funding.py` | 9 | 8h settlements on their own clock |
| `price_ladder.py` | 8 | `windowed_realized_vol`, `90D` windows |
| `basis.py` | 6 | 1m premium index |
| `term_basis.py` | 5 | delivery klines, wall-clock |
| `bipower.py` | 2 | *"**wall-clock windows** at each bar close"* |
| `hurst.py` | 2 | rolling MSD over return lags |
| `perp_spot_volume.py` | 2 | volume ratios over time windows |
| `skew.py` | 2 | `windowed_sum(m, r, t, W)` — wall-clock |
| `co_movement.py` | 1 | *"Computed on the index's **1-minute grid**"* |
| `anatomy.py` | 15 → 11 | return-based; **but see caveat** |

**Caveat on `anatomy.py`.** Its lag features (`bar_return_lag_1/2/5`) are
expressed in *bar* units. At ~65 bars/day on the 200M threshold a lag of 1 was
~22 minutes; on the 8h grid it is 8 hours. The features compute fine and mean
something entirely different. Admitted **only with the lags reinterpreted as
the 8/16/40h ladder**, and recorded as redefined rather than ported.

> **CORRECTED 2026-08-19 — this audit admitted `anatomy.py` wholesale, and
> four of its features should have been excluded.**
>
> The caveat above reinterpreted the *lags* and stopped there. But the same
> module's `bar_duration_*` features measure **how long a dollar bar took to
> fill**, which is the sampling clock itself. On a fixed 8h grid duration is
> a constant, so these are not "clock-dependent" in the sense the lags were —
> they are clock-*defined*, and there is nothing left to reinterpret. They
> fail the audit's own criterion more plainly than the features it caught.
>
> `anatomy.py` therefore contributes **11**, not 15, and the admitted total is
> **61**, not 65. The error is recorded rather than silently edited: it was
> found by the Family I coverage audit, not by the audit that made it, which
> is evidence about how much weight a single-pass clock review can carry.

> **MEASURED 2026-08-10 — the per-coin transforms carry ZERO ranking
> information, and are not built.**
>
> `xsec_rank`, `xsec_demean` and `minus_btc` each preserve within-bar ordering
> **exactly**: rank correlation **1.0000000000 in 100% of bars**. Each is either
> a monotone map (`rank`) or subtracts a constant shared by every coin in the
> bar (`− bar mean`, `− BTC`). Neither can reorder a cross-section, so under a
> ranking objective their Spearman IC is identical to the base feature **by
> construction, not approximately**.
>
> Combined with the three bar-level transforms already known to have zero
> ranking power, **all six of `relative.py`'s transforms collapse.** The
> inherited library's "~260 columns" is really its ~65 base features, and the
> old project's 230-feature set contained far fewer distinct *orderings* than
> its count implied.
>
> A rank transform can still help a tree numerically — different split points,
> robustness to outliers. That is a modelling choice like winsorisation, not a
> feature, and it does not belong in the multiple-testing count.

### 4.2 The relative transforms — and a discovery

`mft/panel/relative.py` generates six variants per base feature. **Three are
per-coin; three are bar-level:**

| transform | per-coin? | ranking power |
|---|---|---|
| `{feat}_xsec_rank` | yes | full |
| `{feat}_xsec_demean` | yes | full |
| `{feat}_minus_btc` | yes | full |
| `mkt_{feat}_disp` | **no** | **zero** |
| `mkt_{feat}_breadth` | **no** | **zero** |
| `mkt_btc_{feat}` | **no** | **zero** |

Bar-level transforms take the **same value for every coin in a
cross-section**, so under `lambdarank` they cannot affect within-bar ordering
at all.

**The old project shipped two of them in its 12-feature "robust" set** —
`mkt_funding_rate_annualized_disp` and `mkt_realized_vol_24h_disp`
(`config/feature_set_robust.txt`). Under its *regression* objective they could
still act through interactions, so it was not a bug there. Under a ranking
objective it would be a straightforward error, and the effective
ranking-feature count was overstated either way.

### 4.3 Treatment

- **Per-coin transforms** — the 65 admitted base features plus their three
  per-coin variants, roughly **260 columns**. Enter the funnel at Stage 0 as
  ordinary candidates.
- **Bar-level transforms** — routed to a separate **context bucket**, capped
  at **5 after clustering**, usable only as interaction terms. They do **not**
  count against the ranking-feature budget, and they are exempt from Stage 1's
  within-bar variation check, which would otherwise delete them wholesale.

### 4.4 What this demands of the funnel

**~330 candidates (67 designed + ~260 inherited) must reduce to ~40.**
Clustering (Stage 4) does most of the work: `open_interest.py` alone
contributes 13 base × 4 = **52 columns representing one mechanism**, not 52
candidates.

If clustering cannot achieve roughly a 10:1 reduction, the funnel has failed
and the list must be cut by hand before target contact — not after.

### 4.5 A caution on reuse

These modules were written for the **200M dollar-bar clock** with `bar_id`
plumbing, which this project abandoned in favour of the 8h wall-clock grid
(`TARGET_DESIGN.md` §4). The formulas and domain knowledge transfer; the
harness does not. Treat the library as a head start on construction, **not as
free features**.

---

## 5. Multiple-testing accounting

**Correction is over MECHANISM FAMILIES, not columns.** This is what makes the
inherited library affordable: 96 VPIN columns are one mechanism, not 96 tests.

Seven families: A Carry, B Positioning, C Microstructure, D Path/attention,
E Return/regime, F Relational, G Spot-perp/basis/premium.

- 7 families → Bonferroni α = 0.05/7 → **z = 2.69**
- ~330 columns would give z ≈ 3.9, which exceeds the benchmark signal's own
  3.6σ and would reject everything including the benchmark — punishing
  thoroughness rather than fishing

Established beforehand, from `TARGET_DESIGN.md` Step 9D:

| quantity | value |
|---|---|
| IC standard error (null) | **0.00366** |
| 2σ threshold | 0.0072 |
| **Family threshold (z = 2.69)** | **0.0099** |
| Benchmark: reversal alone | **0.0133** |

---

## 6. Decision rules, stated in advance

1. **The set must beat the benchmark, not zero.** A feature set scoring 0.015
   has added almost nothing over one parameter-free signal. The comparison is
   against E1 alone.
2. **Report every feature's IC with an interval.** No point estimates.
3. **Report all folds together**, never the best.
4. **A family with no feature clearing z = 2.64 is reported as a null result**
   for that mechanism. Nulls are findings; Family B is the likely candidate
   and that is fine.
5. **Feature count is not a success metric.** Fewer surviving features with a
   higher combined IC beats more features with the same IC.

---

## 7. Deliberately excluded, and why

| excluded | reason |
|---|---|
| Session (raw) | bar-level — identical for every coin, so **zero** ranking power under `lambdarank` |
| Market dispersion, market funding, avg pairwise correlation | same — bar-level. Usable only as interaction terms |
| Coin age | per-coin but nearly constant; a static tilt with no timing |
| Correlation term structure | **measured and rejected** — persistence +0.038, i.e. noise |
| Rolling `corr(coin, BTC)` at 1h | **measured and rejected** — 95% redundant with E6 |
| Liquidation features | data not downloaded |
| aggTrades microstructure | data not downloaded; C3/C4 are the free proxies |
| Semi-beta | **target work, not feature work** — see `TARGET_DESIGN.md` §9 item 8 |
| **A3 `funding_clamp_frac`** | **REMOVED at Stage 1, 2026-08-07.** Clamp reached in 0.029% of settlements (~1 in 3,400), so it is zero for every coin in ~97% of bars — measured within-bar variation 9%. Also the cap is not constant: SOLUSDT reaches 0.0200 vs the 0.0075 standard. Its continuous form is A1 `funding_z`, so a redefinition would be redundant, not additive |
| `vpin.py` (24), `spot_flow.py` (12) | **clock-excluded** (§4.1) — the dollar bars *are* the construction. Mechanisms survive as C6 and G4, rebuilt for this clock |
| Raw session | bar-level; E14 `session_rel_volume` is the per-coin form |
| G11 `term_basis_slope` — **conditional** | delivery futures exist for BTC/ETH only. Per-coin it is NaN for 18 of 20 coins and will fail Stage 0's per-coin coverage floor. Retained so the failure is *recorded* rather than assumed; expected to be routed to the context bucket |

---

## 8. Known risks in this list

1. **Family B priors are weak.** Three of five marked low confidence.
2. **Sector families are small.** Majors and Payments have two members each, so
   F1 collapses to a pair spread for those coins — a legitimate feature but a
   different one, and it should not be pooled silently with the 7-member L1
   sector.
3. **C1 may be dead on arrival.** In a universe this liquid, zero-return
   minutes may be zero everywhere. One query settles it, at Stage 0.
4. **E4 and E5 are products of two noisy quantities**, so they are noisier than
   either input. Stage 2's SE check applies.
5. **The universe is survivorship-biased** in every period, which no feature
   here addresses and which inflates everything.

---

## 9. Amendment policy

Any feature added after 2026-08-07 must be recorded here with its date and
reason, and **counted as an additional test**. Any change to the sector map
must be recorded the same way.

The purpose is not bureaucracy. It is that the previous incarnation of this
project produced conclusions that did not survive contact with the holdout,
and the single cheapest defence is a list written down before anyone knew
which entries would look good.

### Log

**2026-08-07 — Family A built. A3 removed by Stage 1.**

Removed at a **target-free** gate, so the removal costs nothing and biases
nothing — this is the funnel working as designed rather than a retraction.
Family A now carries four features, all gated and written.

Two build defects were found and fixed before anything was kept, both worth
recording because they would have been invisible in the output:

1. **Funding stamps jitter by seconds.** Only 53.3% of settlements match a grid
   instant exactly; 100% fall within one minute. Exact-index alignment silently
   doubled the index and destroyed every rolling window. Fixed with an as-of
   join plus a 9h staleness rejection.
2. **The per-coin coverage gate used the wrong denominator.** It measured
   against all grid instants, so SUIUSDT — which listed in 2023-05 and is
   present in 39% of them — could never reach the 80% floor. Features are now
   restricted to `(t_obs, symbol)` pairs that exist in the decision grid: a
   coin that had not listed is not a missing value, it is not an opportunity.
   Same class of error as the Gate 9A slope threshold.

**2026-08-07 — Amendment 1, same day as the freeze, before any target contact.**

Three changes:

1. **Family G (spot vs perp), 5 features added.** Seam 5 was listed as bet #2
   in `FEATURE_EXPLORATION.md` §13 and was omitted from the original freeze by
   oversight, not judgment.
2. **C6 `vpin_perp_z` added.** VPIN was the single largest block in the old
   project's 88-feature selection and had no representation in the original
   list. Kyle's lambda and Roll's spread are related but measure impact, not
   flow toxicity.
3. **Part 2 added — the full inherited library** (~400 per-coin columns plus a
   capped context bucket), at the user's direction.

**Why this is statistically free.** Pre-registration exists to prevent
additions *informed by results*. No feature has been computed, no IC measured,
no target contact made. Nothing here can be biased by an outcome. The
amendment is recorded because the log is the point, not because the addition
carries a cost.

Family count moved 6 → 7, so the threshold moved 0.0097 → 0.0099.
