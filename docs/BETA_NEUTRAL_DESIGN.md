# Beta-neutral base — design

*Drafted 2026-08-05. Scope is deliberately narrow: this document specifies only
the **base of beta-neutrality** for a cross-sectional crypto strategy. The
target definition, feature universe, label horizon, and model are explicitly
OUT of scope and are decided later, on top of this.*

Status of each decision is marked **DECIDED**, **OPEN**, or **TO TEST**. Nothing
here has been implemented.

---

## 1. Purpose

Build a cross-sectional strategy whose P&L does not depend on the direction of
the crypto market. Everything in this document exists to make that claim
*measurable* rather than assumed.

The rationale for neutrality, on the record so it is not re-litigated later:

- Crypto is dominated by one common factor (PC1 typically explains 70–85% of
  return variance across majors). Left unconstrained, market direction swamps
  cross-sectional signal.
- At crypto volatility, "a small beta" is not small. With BTC annualized vol
  ~50–60%, a book beta of 0.2 contributes ~11% annualized volatility — roughly
  the entire risk budget of a cross-sectional book. The intuition that a little
  beta is harmless is imported from equities, where the same beta costs ~3%.
- **Neutrality is reversible; contamination is not.** A beta-neutral book can be
  given +0.3 beta later with one deliberate overlay position. A signal with beta
  baked into its target and feature selection can never be decomposed again.

Beta exposure, if ever wanted, is a portfolio-level decision made *after* this
base — never a property inherited from the signal.

---

## 2. Foundation

Everything derives from one decomposition:

```
r_i,t  =  α_i,t  +  β_i,t · r_m,t  +  ε_i,t
```

Beta-neutral means the portfolio's loading on `r_m` is zero, so realized P&L is
driven only by `α + ε`. Each decision below makes one term well-defined and
measurable.

---

## 3. D1 — Definition of the market factor `r_m`  **DECIDED**

**Decision: `r_m` is the volume-weighted index of the tradeable universe.**

Beta cannot be estimated before the market is defined, and every downstream
quantity (β, residuals, "neutral") is defined relative to this choice. If it is
wrong, the book is neutral to the wrong thing while carrying hidden exposure.

Rationale: the strategy is exposed to the common factor *of the coins it
actually holds*. BTC alone is a proxy that misses the alt-specific common
factor. An equal-weighted mean overweights the smallest, noisiest names.

Rejected alternatives and why:

| Alternative | Rejected because |
|---|---|
| BTC return | Misses the alt-specific common factor |
| Equal-weighted universe mean | Overweights small coins; noisier |
| PC1 as the traded definition | Estimated, unstable, sign-flips, hard to interpret |

**PC1 is retained as a diagnostic, not as the definition.** The volume-weighted
index is validated by showing it captures most of what PC1 captures, rather than
by assertion. See D2.

Index weights must be **point-in-time**. A weight series computed with
present-day knowledge is lookahead.

---

## 4. D2 — How many factors is the book neutral to?  **TO TEST**

The hazard this addresses: if crypto has a meaningful **second** common factor
(plausibly a BTC-vs-alt or large-vs-small-cap dimension) and the book is
neutralized only to a single market index, the book is *not* neutral. It has had
factor 1 removed and carries an unhedged systematic bet on factor 2 — which will
present as alpha until the factor turns.

### Pre-registered test

Run on the **train split only** (2020-01-01 → 2024-12-27):

1. Compute the correlation matrix of coin returns (frequency per §6).
2. Take its eigenvalues; report the variance share of PC1, PC2, PC3.
3. Report the correlation between PC1 and the volume-weighted index return.

### Pre-registered decision rule, with expected result

Stated before running, so it cannot be retrofitted:

- **If PC2 variance share ≥ 15%** → adopt two-factor neutralization; the base
  becomes `Σ w·β⁽¹⁾ = 0` and `Σ w·β⁽²⁾ = 0`.
- **If PC2 < 15%** → single-factor neutralization is sufficient; record the
  number and move on.
- **If corr(PC1, volume-weighted index) < 0.90** → the D1 choice is not
  capturing the dominant factor and D1 must be revisited.

*Expected (prior, recorded in advance): PC1 ≈ 70–85%, PC2 ≈ 5–12%,
corr(PC1, index) > 0.95. If the outcome is far from this, that is itself the
finding and warrants investigation before proceeding.*

**No work proceeds past this test.** The number determines whether the rest of
the design is single- or two-factor.

---

## 5. D3 — Where neutrality is enforced  **DECIDED**

Three distinct places, commonly conflated. They are independent choices:

| Layer | What it does | Decision |
|---|---|---|
| **Construction** | Weights satisfy `Σ w·β = 0` | **Mandatory** |
| **Target** | Label is residualized: `r_i − β_i·r_m` | Recommended, but must be earned (§9) |
| **Features** | Inputs residualized before use | **Not adopted** — screen instead |

**Only construction-level neutrality makes the book neutral.** This is the
central point. A perfectly idiosyncratic signal can still produce a beta-loaded
book if the names it happens to select have asymmetric betas across the legs —
nothing about a residualized *target* constrains the *portfolio*.

Feature-level residualization is rejected because it destroys information
indiscriminately. Instead, features are **screened** for beta-proxying via the
regime split in §8.

### Construction mechanism

Dollar-neutral + beta-neutral + a desired weighting scheme is three constraints
that generally cannot be satisfied simultaneously. Resolution: take the desired
weights `w₀` and **project** them onto the constraint set — the nearest weights
that satisfy both constraints.

Minimize `‖w − w₀‖²` subject to `Σw = 0` and `Σ w·β = 0`.

With `A = [1, β]ᵀ` (a 2×n constraint matrix), the closed-form projection is:

```
w  =  w₀ − Aᵀ (A Aᵀ)⁻¹ A w₀
```

A 2×2 inverse; deterministic, no optimizer, no tuning. "Nearest to `w₀`" keeps
the intended sizing logic intact while making the constraints exact. Extends
directly to two factors by adding a row to `A`.

---

## 6. Beta estimation

`β̂` is the quantity everything now depends on, and the most likely source of
failure. Four requirements:

**Point-in-time — non-negotiable.**  `β̂_i,t` uses only data before `t`, via a
rolling or expanding window. A beta fitted on the full sample injects future
information directly into the label and the weights. This must be enforced by an
assertion, not a convention.

**Return frequency: NOT bar-level.**  **OPEN** — resolve empirically.
Non-synchronous trading biases high-frequency correlations toward zero (the Epps
effect), so betas estimated on ~26-minute bars will be systematically too low —
and *differentially* so, worst for the least-traded coins, which is precisely a
cross-sectional distortion. Estimate beta on aggregated returns (1h–4h
candidates) even though the strategy trades on bars. Resolve by computing β at
several frequencies and adopting the point where estimates stabilize.

**Window length.**  **OPEN.** Too short is noise; too long is stale, and crypto
betas genuinely move. Choose on economic grounds and record the reasoning — this
is a lookback choice carrying the same multiple-testing exposure as any other,
and it is not to be swept for best backtest performance.

**Shrinkage.**  **OPEN** (method), **DECIDED** (that it is applied). Raw rolling
betas across 20 crypto coins are noisy, and noise in `β̂` propagates straight
into both the label and the weights. Shrink toward 1.0 — crypto betas cluster
near 1, so the prior is well founded and costs little when the raw estimate is
good. Vasicek (precision-weighted) or Blume-style are both candidates.

---

## 7. D4 — Ex-ante vs ex-post neutrality  **DECIDED**

The distinction where beta-neutral strategies actually fail:

- **Ex-ante beta** — what was constructed, using `β̂`. Zero by construction.
- **Ex-post beta** — regress *realized* book returns on *realized* market
  returns. The slope is the actual exposure that was carried.

These differ, always, because `β̂` carries estimation error and true beta drifts
between estimation and holding. **Both must be reported with every result.**

A book that is ex-ante neutral and ex-post 0.25 is not a market-neutral
strategy; it is a directional one with extra steps. The gap between the two is
the honest measure of beta-estimation quality and belongs in every report.

---

## 8. Verification requirements

Neutrality is a claim, and every claim here must be checked by something that
halts rather than something that prints.

**Assertions in the inner loop, every rebalance:**

- `|Σ w| < 1e-9` (dollar-neutral)
- `|Σ w·β̂| < 1e-9` (beta-neutral, ex-ante)
- Legs disjoint — no coin in both
- `β̂` timestamps strictly precede the decision bar (point-in-time guard)

**Reported with every result, without exception:**

- Ex-post book beta, with a confidence interval — never a bare point estimate
- The ex-ante / ex-post gap
- Book beta **conditional on regime**, not pooled

**Regime conditioning is mandatory, not optional.** Crypto correlations converge
toward 1 in stress, so a book that is neutral in calm markets can become
directionally exposed exactly when it is most costly. Pooled-average neutrality
hides this completely. At minimum: up-market vs down-market, plus the highest
`|r_m|` decile as a stress subset.

Any feature whose up-market IC and down-market IC differ materially is a beta
proxy regardless of what its construction suggests, and is screened out here
(this replaces feature-level residualization, per D3).

---

## 9. Open questions

Carried forward deliberately rather than guessed:

1. **Factor count** (§4) — gates everything; run first.
2. **Beta estimation frequency** (§6) — resolve via the Epps stability check.
3. **Beta window length** (§6) — economic grounds, recorded.
4. **Shrinkage method and intensity** (§6).
5. **Does target-level residualization actually help?** It replaces a clean
   observable (`r_i`) with an estimated quantity carrying estimation error. If
   `β̂` is noisy enough, it injects more noise than it removes contamination.
   This is a **testable proposition with exactly one variable**: identical
   features and construction, target with and without residualization.
   Pre-register the decision rule before looking.
6. **Residual beta of the universe itself** — with 20 coins and betas clustered
   0.8–1.3, how much freedom does the projection in §5 actually have? If the
   constraint set is nearly binding, neutrality may cost meaningful deviation
   from the intended weights. Worth measuring early.

---

## 10. Method discipline — binding on this work

Carried forward from the post-mortem of the previous pipeline. These are
process rules, not findings, and they apply to everything built on this design:

1. **One variable per arm.** If an arm differs from baseline in two ways, it is
   an anecdote, not an experiment. Decompose first.
2. **Sanity gates ABORT, they do not print.** Any reconstruction or identity
   check halts the run on failure. A diagnostic that is printed and not read is
   not a diagnostic.
3. **Pre-register the decision rule including the statistical test**, and state
   the expected result, so it cannot be retrofitted. §4 is written this way
   deliberately, as the template.
4. **Invariants as assertions** in the inner loop, not comments in a header.
5. **Multiple windows reported together** — never the best one.
6. **Regime split** for anything claiming to be market-neutral. Non-negotiable
   here, since neutrality is the entire product.
7. **Noise floor before point estimates.** Measure seed and bootstrap dispersion
   first; report intervals, not points.
8. **Plausibility check on magnitude.** If an input changed by a hair and the
   output moved a lot, find out why before believing it.

All research runs on the train split only (2020-01-01 → 2024-12-27, 120h
purged). 2025 and 2026 are sealed. See `mft/splits.py`.

---

## 11. What is explicitly NOT decided here

Out of scope for this document, to be designed on top of this base:

- Target definition — horizon, scaling, whether funding is included
- Model objective — ranking vs regression vs classification
- Feature universe
- Universe size and composition
- Position sizing scheme (beyond the requirement that §5 projects it onto the
  constraint set)
- Rebalance cadence
