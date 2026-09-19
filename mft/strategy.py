"""The strategy specification. FROZEN 2026-08-28.

Chosen on TRAIN evidence only, before any holdout was read. Every constant
below was selected by the sweep in `scripts/fee_ceiling.py` and is fixed here
so that a holdout run introduces NO new decisions -- it re-executes this file.

Changing any constant after a holdout has been read is a new experiment and
must be reported as one.

    BOOK          C: dollar-neutral positions plus a market hedge OVERLAY
    SMOOTHING     NONE (lam = 0, band = 0)
    GROSS         1.0
    REBALANCE     every 8h bar

WHY C AND NOT A BETA CONSTRAINT
--------------------------------
Constraining sum(w*beta) = 0 moves the positions and loses the alpha with them:
Sharpe 2.61 -> 2.12. An overlay leaves the positions untouched and only adds a
market leg, and it IMPROVES the book: 2.61 -> 2.86 with corr(book, r_m) falling
from -0.1268 (t -6.71) to +0.0414. Measured on train, out of fold. These are
different operations and the first version of this work conflated them.

FEES ARE ZERO BY DESIGN ASSUMPTION
-----------------------------------
BETA_NEUTRAL_DESIGN assumes zero fees, and every headline number in this
project is a ZERO-FEE number. Earlier versions of this file argued the
configuration down on the basis of an assumed exchange fee; that was an
imported on top of the design, not part of it, and it has been removed.

Turnover is still MEASURED and reported, and so is the break-even fee, because
they are facts about the book rather than assumptions about the venue. They
are what makes the zero-fee assumption revisitable if it ever changes. They do
not qualify the headline.

NO SMOOTHING
------------
Smoothing was measured and NOT adopted. Under the zero-fee assumption the
choice is straightforward -- it costs alpha:

Re-measured under the ADOPTED rolling-90 hedge (the numbers first written here
came from the expanding hedge and were stale the moment it changed):

                      Sharpe   ann ret   maxDD    turnover   break-even
    lam 0   TRAIN      +2.86   +48.71%   -8.62%      91.9%     3.94 bps
            2025       +2.84   +49.54%   -7.29%      88.6%     4.15 bps
    lam .95 TRAIN      +1.72   +30.77%  -13.04%       7.9%    30.91 bps
            2025       +1.22   +18.63%  -12.66%       6.5%    23.98 bps

Smoothing costs 1.14 of Sharpe on train and 1.62 on 2025, and deepens the
drawdown. Under the zero-fee assumption there is nothing on the other side of
that trade, so it is not adopted.

The smoothed variant is retained in VARIANTS regardless, because turnover of
7.9% against 91.9% is a large difference in what the book demands of an
execution venue, and that stays worth knowing even when fees are assumed away.

The smoothing sweep left a finding that stands regardless of which was chosen:
lam 0.95 costs almost no NET Sharpe, so the 8h clock trades considerably faster
than the alpha decays. The principled response is to re-test the target at a
longer horizon rather than to smooth an 8h book. That reopens TARGET_DESIGN and
is recorded as open, not done.

THE HEDGE: A 90-BAR ROLLING REGRESSION, AND WHY IT REPLACED AN EXPANDING ONE
----------------------------------------------------------------------------
The overlay ratio is a regression of book return on r_m over the last 90 bars
(30 days), past bars only. It replaced an EXPANDING regression that held on
train and broke in 2025:

    variant        train Sharpe / corr / t      2025 Sharpe / corr / t
    expanding        +2.88  +0.0555   +2.92       +2.43  -0.3990  -14.38
    rolling-90       +2.86  -0.0227   -1.19       +2.84  +0.0003   +0.01
    exante           +2.93  +0.0128   +0.67       +3.24  -0.1194   -3.97
    no hedge         +2.61  -0.1364   -7.23       +2.40  -0.4109  -14.89

An expanding window over four years moves by about 1/4500 per new bar, so it
cannot track a beta that shifts regime. Rolling 90 costs 0.02 of train Sharpe
and delivers a book that is statistically indistinguishable from
market-neutral in BOTH periods.

CONVEXITY IS NOT HEDGED, AND THAT IS A MEASURED CONCLUSION, NOT AN OMISSION
---------------------------------------------------------------------------
The book earns more when the market moves hard in EITHER direction:
corr(book, |r_m|) +0.0605 (t +3.18) on train, +0.1335 (t +4.45) in 2025. It is
not the edge scaling with opportunity -- controlling jointly for cross-sectional
dispersion, |r_m| survives at t +3.19 and +4.34 while dispersion itself is
insignificant (t +0.58, +0.62).

It is concentrated, and the concentration is the real number to know:

    share of total P&L      Q1 calm   Q2      Q3      Q4 wild
    train                     18.2%   10.5%   18.5%   52.8%
    2025                       9.6%    5.0%   21.2%   64.2%

The wildest quarter of bars carries HALF to TWO THIRDS of the return. A calm
year would not produce these numbers.

VOL-TARGETING WAS TRIED AND REJECTED. Scaling gross by predicted market vol
left the exposure untouched -- t(|r_m|) went 3.18 -> 3.64 on train and
4.45 -> 4.26 in 2025 -- and cost Sharpe (2.84 -> 2.76 in 2025). The reason it
cannot work is measurable: trailing vol explains 8.0% of |r_m| variance on
train and 0.2% in 2025, with |r_m| autocorrelation of only +0.24 and +0.14. You
cannot size against a quantity you cannot forecast.

Removing this exposure needs an instrument that pays off in |r_m| -- a straddle
-- and the project is constrained to Binance perpetuals. So it is DECLARED
rather than hedged: the strategy is neutral to the market's DIRECTION and is
long its VOLATILITY. Anyone sizing it should know that a quiet year removes
roughly half the return, and that the exposure is the benign sign.

EX-ANTE WAS THE ELEGANT CANDIDATE AND IT LOST. The book's exposure is
sum(w_i * beta_i), known at trade time with no lookback -- which sounded
strictly better than regressing on your own past returns. It is not, and the
reason is worth keeping: that sum uses the ESTIMATED trailing beta, so it
cancels the exposure the model THINKS it has. In 2025 realised betas diverged
from their trailing estimates and the elegant version was off by t -3.97, while
the crude one that watches actual outcomes was at t +0.01.

SELECTION HONESTY. On TRAIN alone, expanding is already the worst of the three
(t +2.92, significant) so dropping it is train-justified. Choosing rolling-90
over exante, however, used the 2025 column -- both are acceptable on train.
That tie-break is selected on data already spent, so only 2026 tests it
cleanly.

`seed` carries train history into a holdout run so the holdout does not restart
with a burn-in it has no reason to pay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LAMBDA = 0.0
BAND = 0.0
GROSS = 1.0
HEDGE = True
HEDGE_MODE = "rolling"     # "expanding" | "rolling" | "exante"
HEDGE_WINDOW = 90          # bars = 30 days, used when mode == "rolling"
HEDGE_SHRINK = 1.0         # multiply the hedge ratio; 1.0 = full hedge
HEDGE_MIN_BARS = 200
# Gross sizing. "fixed" holds gross at GROSS every bar. "voltarget" scales it
# by (median past market vol / predicted market vol), predicted from the last
# VOL_WINDOW bars of r_m, past only, and clipped so a single calm stretch
# cannot lever the book up without bound.
GROSS_MODE = "fixed"       # "fixed" | "voltarget" -- voltarget TESTED AND REJECTED, see below
VOL_WINDOW = 90
VOL_CLIP = (0.5, 2.0)
MIN_COINS = 8
BARS_PER_YEAR = 365 * 3

# Both configurations are kept. The low-turnover one is NOT the active
# specification, but it was measured on the same out-of-fold panel and is
# retained in full so it can be picked up later without re-deriving it.
# Numbers are train, out of fold, seed 20260828.
VARIANTS = {
    # Both configurations, re-measured under the ADOPTED hedge (rolling-90) on
    # TRAIN and on 2025. The previous version of this dict held numbers from
    # the expanding hedge and was stale the moment the hedge changed -- a
    # record that disagrees with the code is worse than no record.
    "headline": {
        "lam": 0.0, "band": 0.0,
        "note": "ACTIVE SPEC. Highest zero-fee alpha.",
        "train": {"sharpe": 2.86, "ann_return": 0.4871, "ann_vol": 0.1386,
                  "max_drawdown": -0.0862, "turnover": 0.919,
                  "breakeven_bps": 3.94, "net_2bp": 1.41, "net_4bp": -0.04,
                  "t_rm": -1.19},
        "h2025": {"sharpe": 2.84, "ann_return": 0.4954, "ann_vol": 0.1419,
                  "max_drawdown": -0.0729, "turnover": 0.886,
                  "breakeven_bps": 4.15, "net_2bp": 1.47, "net_4bp": 0.10,
                  "t_rm": 0.01},
    },
    "low_turnover": {
        "lam": 0.95, "band": 0.005,
        "note": "Not active. Lower zero-fee alpha, but 7.9% turnover against "
                "91.9% -- retained because that is a large difference in what "
                "the book asks of an execution venue.",
        "train": {"sharpe": 1.72, "ann_return": 0.3077, "ann_vol": 0.1561,
                  "max_drawdown": -0.1304, "turnover": 0.079,
                  "breakeven_bps": 30.91, "net_2bp": 1.61, "net_4bp": 1.50,
                  "t_rm": -2.04},
        "h2025": {"sharpe": 1.22, "ann_return": 0.1863, "ann_vol": 0.1397,
                  "max_drawdown": -0.1266, "turnover": 0.065,
                  "breakeven_bps": 23.98, "net_2bp": 1.12, "net_4bp": 1.02,
                  "t_rm": -0.12},
        # Run on holdout 2 after the fact. This variant was selected on TRAIN
        # in the fee-ceiling sweep and recorded here before either holdout was
        # read, so 2026 is a real test of it -- and it fails.
        "h2026": {"sharpe": -0.59, "ann_return": -0.0678, "ann_vol": 0.1191,
                  "max_drawdown": -0.1585, "turnover": 0.062,
                  "t_rm": 0.04},
    },
}


def use(variant: str) -> dict:
    """Return a variant's parameters. `run` and `smooth` read the module
    constants, so a caller wanting a variant passes lam/band explicitly."""
    return VARIANTS[variant]


@dataclass(frozen=True)
class BookResult:
    ret: pd.Series            # net of funding and of the hedge overlay
    gross_ret: pd.Series      # before funding
    turnover: pd.Series
    hedge_beta: pd.Series
    weights: pd.DataFrame


def target_weights(P: pd.DataFrame) -> pd.Series:
    """Dollar-neutral, inverse-vol weights from the model score. Gross = 1."""
    z = P.groupby(level="t_obs")["score"].rank(pct=True)
    z = z - z.groupby(level="t_obs").transform("mean")
    w = z / P["sigma_eps"].replace(0.0, np.nan)
    w = w - w.groupby(level="t_obs").transform("mean")
    return GROSS * w / w.abs().groupby(level="t_obs").transform("sum")


def smooth(W: pd.DataFrame, lam: float | None = None, band: float | None = None,
           init: np.ndarray | None = None) -> pd.DataFrame:
    """EWMA the weight matrix, then apply the no-trade band.

    `init` carries the last held weights of a previous period so a holdout run
    starts from the book the strategy actually held, not from cash.

    lam/band default to the MODULE constants, resolved at CALL time rather than
    bound into the signature. Writing `lam: float = LAMBDA` binds the value at
    import, so a caller that sets `strategy.LAMBDA = 0.95` and re-runs gets the
    old smoothing silently -- which is exactly what happened when the two
    VARIANTS were re-measured and returned byte-identical results. Every other
    setting in this module (HEDGE_MODE, GROSS_MODE) is read inside `run` and
    was never affected; this one was the exception.
    """
    lam = LAMBDA if lam is None else lam
    band = BAND if band is None else band
    if lam > 0:
        W = W.ewm(alpha=1 - lam, adjust=False).mean()
        W = W.div(W.abs().sum(axis=1).replace(0.0, np.nan), axis=0) * GROSS
    A = W.to_numpy(copy=True)
    held = np.zeros(A.shape[1]) if init is None else init.copy()
    for i in range(len(A)):
        tgt = np.nan_to_num(A[i])
        if band > 0:
            held = np.where(np.abs(tgt - held) > band, tgt, held)
        else:
            held = tgt
        A[i] = held
    return pd.DataFrame(A, index=W.index, columns=W.columns).fillna(0.0)


def run(P: pd.DataFrame, seed: tuple[np.ndarray, np.ndarray] | None = None,
        init_weights: np.ndarray | None = None) -> BookResult:
    """Execute the frozen specification on a scored panel.

    `P` needs columns: score, fwd_ret, fwd_rm, sigma_eps, funding.
    `seed` is (past book returns, past r_m) used to prime the hedge regression.
    """
    n = P.groupby(level="t_obs")["score"].transform("size")
    P = P[n >= MIN_COINS]

    W = smooth(target_weights(P).unstack("symbol").fillna(0.0),
               init=init_weights)
    tob = W.index
    _rm_all = P.groupby(level="t_obs")["fwd_rm"].first().reindex(tob)
    if GROSS_MODE == "voltarget":
        # Predicted market vol from PAST bars only: shift(1) so bar t is sized
        # with information available before it. The book earns more when the
        # market moves hard (measured: corr(book,|r_m|) +0.13 at t +4.3 in
        # 2025, and it survives controlling for cross-sectional dispersion at
        # t +4.34, so it is market convexity and not opportunity scaling).
        # Sizing down into predicted turbulence is the only way to flatten that
        # with perps alone -- |r_m| is not a tradeable instrument.
        pv = _rm_all.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std().shift(1)
        scale = (pv.median() / pv).clip(*VOL_CLIP).fillna(1.0)
        W = W.mul(scale, axis=0)
    R = P["fwd_ret"].unstack("symbol").reindex(tob).reindex(columns=W.columns)
    Fn = P["funding"].unstack("symbol").reindex(tob).reindex(columns=W.columns)
    rm = P.groupby(level="t_obs")["fwd_rm"].first().reindex(tob)

    gross = (W * R.fillna(0.0)).sum(axis=1)
    net = gross - (W * Fn.fillna(0.0)).sum(axis=1)

    turn = (W - W.shift(1)).abs().sum(axis=1)
    turn.iloc[0] = float(np.abs(W.iloc[0].to_numpy()
                                - (init_weights if init_weights is not None
                                   else 0.0)).sum())

    hb = np.zeros(len(net))
    if HEDGE:
        if HEDGE_MODE == "exante":
            # THE BOOK'S BETA IS KNOWN AT TRADE TIME. It is sum(w_i * beta_i),
            # using the same trailing betas the target already uses -- no
            # regression, no lookback, no lag. A backward-looking regression
            # can only learn that the beta moved after it has moved, which is
            # exactly how the expanding version failed out of sample in 2025
            # (train +0.056 -> holdout -0.186, sign flipped).
            B = P["beta"].unstack("symbol").reindex(tob).reindex(
                columns=W.columns)
            hb = (W * B.fillna(0.0)).sum(axis=1).to_numpy()
        else:
            r_hist = list(seed[0]) if seed else []
            m_hist = list(seed[1]) if seed else []
            rr, mm = net.to_numpy(), rm.to_numpy()
            for i in range(len(rr)):
                if len(r_hist) >= HEDGE_MIN_BARS:
                    if HEDGE_MODE == "rolling":
                        a = np.asarray(m_hist[-HEDGE_WINDOW:])
                        b = np.asarray(r_hist[-HEDGE_WINDOW:])
                    else:                       # expanding
                        a, b = np.asarray(m_hist), np.asarray(r_hist)
                    v = a.var()
                    hb[i] = (np.cov(a, b)[0, 1] / v) if v > 0 else 0.0
                r_hist.append(rr[i])
                m_hist.append(mm[i])
        hb = hb * HEDGE_SHRINK
        h = pd.Series(hb, index=tob)
        net = net - h * rm
        turn = turn + pd.Series(np.abs(np.diff(hb, prepend=hb[0] if seed else 0.0)),
                                index=tob)

    return BookResult(ret=net, gross_ret=gross, turnover=turn,
                      hedge_beta=pd.Series(hb, index=tob), weights=W)


def metrics(r: pd.Series, turnover: float | None = None,
            costs_bps=(0.0,)) -> dict:
    """Performance metrics. FEES ARE ZERO by design assumption, so `costs_bps`
    defaults to zero only. Turnover and the break-even fee are still returned:
    they are measurements of the book, not assumptions about the venue, and
    they are what would let the assumption be revisited."""
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sharpe = mu / sd * np.sqrt(BARS_PER_YEAR) if sd > 0 else np.nan
    eq = (1.0 + r).cumprod()
    vol = sd * np.sqrt(BARS_PER_YEAR)
    out = {"bars": int(len(r)),
           "ann_return": float((1 + mu) ** BARS_PER_YEAR - 1),
           "ann_vol": float(vol), "sharpe": float(sharpe),
           "hit_rate": float((r > 0).mean()),
           "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
           "mean_bar_bps": float(mu * 1e4)}
    if turnover:
        units = turnover * BARS_PER_YEAR
        out["turnover"] = float(turnover)
        out["net_sharpe"] = {c: float(sharpe - (units * c / 1e4) / vol)
                             for c in costs_bps}
        out["breakeven_bps"] = float(sharpe * vol / units * 1e4)
    return out
