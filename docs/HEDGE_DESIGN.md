# The hedge overlay — how it was chosen, and how to read the table

## 1. The table

Measured by `scripts/fix_hedge.py`. Train is 2,755 out-of-fold bars
(2022-06 → 2024-12); 2025 is 1,094 bars, the spent holdout.

| variant | train Sharpe / corr / t | 2025 Sharpe / corr / t |
|---|---|---|
| expanding *(shipped)* | +2.88 / +0.056 / **+2.92** | +2.43 / −0.399 / **−14.38** |
| **rolling-90** *(adopted)* | +2.86 / −0.023 / **−1.19** | +2.84 / +0.0003 / **+0.01** |
| rolling-270 | +2.73 / −0.031 / −1.64 | +2.83 / −0.122 / −4.05 |
| ex-ante `Σwβ` | +2.93 / +0.013 / +0.67 | +3.24 / −0.119 / −3.97 |
| no hedge | +2.61 / −0.136 / −7.23 | +2.40 / −0.411 / −14.89 |

---

## 2. What each number is

### Sharpe

Annualised return divided by annualised volatility — return per unit of risk.
Zero fees. It is **not** the criterion here; see §4.

### corr

The correlation between the book's 8h return and the **market's** 8h return
(`r_m`), across every bar in the period.

- **0** — the book's P&L is unrelated to which way the market went. This is
  what "market-neutral" means, and it is the whole objective of the overlay.
- **negative** — the book makes money when the market falls. It is
  accidentally short.
- **positive** — the book makes money when the market rises. It is
  accidentally long.

Either sign is a failure. A dollar-neutral book is *supposed* to be indifferent.

### t — the t-statistic

**`corr` alone cannot tell you whether an exposure is real.** A correlation
measured over a finite number of bars is never exactly zero even when the true
exposure is zero, purely from noise. `t` answers the question *corr* cannot:

> Is this correlation far enough from zero that noise cannot explain it?

$$t = \frac{\text{corr} \times \sqrt{n-2}}{\sqrt{1-\text{corr}^2}} \qquad \approx \ \text{corr} \times \sqrt{n}$$

It is the correlation expressed **in units of its own standard error**. `t = 3`
means the correlation sits three standard errors from zero.

**Reading it:**

| \|t\| | meaning |
|---|---|
| < 2 | indistinguishable from zero — **neutral** |
| 2–3 | probably real |
| > 3 | real |
| > 10 | not remotely neutral |

The conventional cut is **\|t\| = 2**, which corresponds to roughly a 5% chance
of seeing a correlation this large if the true exposure were zero.

**Why `t` and not `corr`.** Sample size changes what a given correlation means:

- train has 2,755 bars → `t ≈ corr × 52`
- 2025 has 1,094 bars → `t ≈ corr × 33`

So the same −0.03 correlation is `t ≈ −1.6` on train and `t ≈ −1.0` on 2025 —
neutral in both. But −0.12 is `t ≈ −4` on 2025, which is not. Comparing raw
correlations across periods of different length would be comparing two
different things; comparing `t` is comparing like with like.

*Worked example, rolling-90 on 2025:* corr = +0.0003, n = 1,094.
`t = 0.0003 × √1092 / √(1−0.0003²) = +0.01`. The book's return over 2025 is
statistically indistinguishable from having no market exposure at all.

---

## 3. Reading each row

**expanding — the shipped version, and why it failed.**
On train it was already marginal: `t +2.92` is *above* the significance cut, so
the book was significantly, if slightly, long the market. Out of sample it
collapsed to `t −14.38`. The ratio was an expanding regression over four years,
so by 2025 a single new bar moved it by about 1/4500. A regression that slow
cannot track a beta that changes regime, and it only ever learns that the beta
moved *after* it has moved.

**rolling-90 — adopted.** The same regression over the last 90 bars (30 days).
Neutral in both periods (`−1.19`, `+0.01`) at a cost of 0.02 Sharpe on train.

**rolling-270.** Neutral on train, `t −4.05` in 2025. Still too slow; 90 days
of memory is more than this beta holds still for.

**ex-ante `Σwβ` — the elegant candidate that lost.** The book's exposure is
`Σ wᵢβᵢ`, computable at trade time with no lookback, so it *should* beat
regressing on your own past returns. It does not, and the reason is worth
keeping: that sum uses the **estimated trailing β**, so it cancels the exposure
the model *thinks* it has. When realised betas diverged from their trailing
estimates in 2025 it landed at `t −3.97`, while the crude method that watches
actual outcomes reached `t +0.01`.

**no hedge.** The raw exposure, showing what the overlay is correcting:
`t −7.23` on train and `t −14.89` in 2025. This is the leak described in
`strategy.py` — dollar-neutrality does not imply beta-neutrality, because
inverse-vol sizing systematically overweights low-β coins.

---

## 4. Sharpe is not the criterion, and one row shows why

Sharpe barely moves across the table — +2.61 to +3.24 — while `t` moves from
+0.01 to −14.89. **The hedge choice is about neutrality, not return.**

Note `ex-ante` has the *highest* 2025 Sharpe, +3.24. It also has `corr −0.119`
in a year when the market index fell 11.3%. A book accidentally short a falling
market earns money for it, and some of that +3.24 is that bet rather than
alpha. Ranking these variants by Sharpe would have selected the one carrying
the largest uncontrolled directional position — which is precisely the failure
the overlay exists to prevent.

The selection rule is therefore: **smallest worst-case |t| across both
periods.**

---

## 5. Selection honesty

2025 has been spent. Any choice informed by the 2025 column is *selected on*
2025, which makes it a validation set for that choice rather than a test.

- Dropping `expanding` is **train-justified**: `t +2.92` on train alone already
  fails, before 2025 is consulted.
- Choosing `rolling-90` over `ex-ante` used the 2025 column — both are
  acceptable on train (`−1.19` vs `+0.67`). **That tie-break is selected on
  spent data, and only 2026 tests it cleanly.**

---

## 6. What the hedge does not fix

The overlay is **linear** — it cancels exposure to `r_m`, the market's
direction. It does nothing about exposure to `|r_m|`, the market's *size of
move*, and was never intended to.

That exposure is real and remains: `corr(book, |r_m|)` = +0.0605 (`t +3.18`) on
train and +0.1335 (`t +4.45`) in 2025. It is not the edge scaling with
opportunity — controlling jointly for cross-sectional dispersion, `|r_m|`
survives at `t +3.19` and `t +4.34` while dispersion is insignificant.

Vol-targeting was implemented and rejected: trailing vol explains 8.0% of
`|r_m|` variance on train and **0.2%** in 2025, so sizing against it added
turnover and subtracted return without moving `t`. Cancelling it needs an
instrument that pays off in `|r_m|` — a straddle — and this project is
constrained to Binance perpetuals.

So the claim is stated narrowly: **the strategy is neutral to the market's
direction and is long its volatility.** The wildest quarter of bars carries
52.8% of train P&L and 64.2% of 2025 P&L. A quiet year removes roughly half the
return.
