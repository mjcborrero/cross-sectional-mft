"""Family H -- the inherited library, reconstructed. FEATURE_LIST_FROZEN.md §4.

The old project's feature code survives in `mft/features/`, but it was written
for a 200M dollar-bar clock with `bar_id` plumbing that this project abandoned.
The formulas transfer; the harness does not. These are RECONSTRUCTIONS on the
8h wall-clock grid, not ports.

WHAT IS HERE, AND WHAT IS NOT

The clock audit (§4.1) admitted 65 of 101 base features. Most of those
mechanisms are already built as designed features with stated priors:

    funding.py (9)          -> Family A          basis.py (6)      -> Family G
    open_interest.py (13)   -> Family B          term_basis.py (5) -> Family G
    bipower.py (2)          -> Family D          perp_spot_vol (2) -> Family G
    skew.py (2)             -> Family D          co_movement.py    -> Family F

Rebuilding those here would add columns that Stage 4 clustering exists to
delete. What remains are the mechanisms with no designed equivalent:
`anatomy.py`'s return-lag ladder, `price_ladder.py`'s momentum/volatility
percentile ladder, and `hurst.py`.

THE RELATIVE TRANSFORMS ARE NOT BUILT, AND THIS IS MEASURED, NOT ASSUMED.
`relative.py` generates six variants per base feature. Three are bar-level and
have zero ranking power (§4.2). The other three -- `xsec_rank`, `xsec_demean`,
`minus_btc` -- preserve within-bar ordering EXACTLY: measured at rank
correlation 1.0000000000 in 100% of bars, because each is either a monotone
map or subtracts a constant shared by every coin in the bar. Under a ranking
objective they carry zero additional information and their Spearman IC is
identical by construction. Building them would triple the multiple-testing
count for no ordering content.

(A rank transform can still help a tree numerically -- different split points,
robustness to outliers. That is a modelling choice like winsorisation, not a
feature, and it does not belong in the count.)

LAGS ARE REDEFINED, NOT PORTED. `anatomy.py`'s `bar_return_lag_1/2/5` were bar
units: at ~65 bars/day a lag of 1 was ~22 minutes. On this grid the same code
means 8 hours. They are rebuilt as an explicit 8/16/40h ladder and recorded as
redefined.

Everything is residualised and idio-vol normalised, per the §1 design
principle -- a raw-return version would carry beta the target has removed, and
would lose to Family E for reasons unrelated to the mechanism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.featbuild import MS_HOUR, Family, GridContext

W30D = 90            # 30 days of 8h steps
MIN30D = 54
W90D = 270
MIN90D = 162
LAGS = {"8h": 1, "16h": 2, "40h": 5}
HURST_LAGS = (1, 2, 4, 8, 16)      # in 8h steps


class Inherited(Family):
    code = "H"
    name = "Inherited library (reconstructed on the 8h clock)"
    # Transient BY CONSTRUCTION -- a difference, a ratio to a lagged
    # value, a deviation from own history, or a windowed return sum.
    # Not stable coin characteristics, so Stage 3 persistence does not
    # apply. Classified from construction, not from any score.
    change_columns = (
        "hurst_change",
        "hurst_short_minus_long",
        "realized_vol_zscore_30d",
        "realized_vol_percentile_90d",
        "resid_sum_5",
        "resid_sum_20",
        "resid_lag_8h",
        "resid_lag_16h",
        "resid_lag_40h",
    )
    warmup_bars = W90D

    def compute(self, ctx: GridContext) -> pd.DataFrame:
        grid = ctx.grid()
        t_obs = ctx.instants()
        idx = pd.Index(t_obs, name="t_obs")

        P = grid.pivot(index="t_obs", columns="symbol", values="px_obs").reindex(idx)
        bi = ctx.artifact("beta_idiovol_8h")
        BETA = bi.pivot(index="t_obs", columns="symbol", values="beta").reindex(idx)
        SIG = bi.pivot(index="t_obs", columns="symbol", values="sigma_eps").reindex(idx)
        wts = ctx.artifact("market_weights_30d")
        W = wts.pivot(index="t_obs", columns="symbol", values="weight") \
               .reindex(idx).reindex(columns=P.columns)

        # 8h residual return, standardised -- the unit everything below uses.
        R = P / P.shift(1) - 1.0
        M = W.shift(1).where(R.notna())
        M = M.div(M.sum(axis=1), axis=0)
        rm = (M * R).sum(axis=1, min_count=1).where(M.notna().sum(axis=1) >= 10)
        e = R.sub(BETA.mul(rm, axis=0)).div(SIG)

        cols: dict[str, pd.DataFrame] = {}

        # --- anatomy.py: the return-lag ladder, REDEFINED in wall-clock -----
        for tag, k in LAGS.items():
            cols[f"resid_lag_{tag}"] = e.shift(k)
        cols["resid_sum_5"] = e.rolling(5, min_periods=3).sum()
        cols["resid_sum_20"] = e.rolling(20, min_periods=12).sum()

        # --- price_ladder.py: momentum and volatility ladders --------------
        mom = e.rolling(3, min_periods=2).sum()          # 24h residual momentum

        rvol = e.rolling(W30D, min_periods=MIN30D).std()
        cols["realized_vol_zscore_30d"] = _z(rvol, W30D, MIN30D)
        cols["realized_vol_percentile_90d"] = _pct(rvol, W90D, MIN90D)

        # --- hurst.py: MSD slope over a lag ladder -------------------------
        c = e.cumsum()
        short = _hurst(c, W30D, MIN30D)
        long = _hurst(c, W90D, MIN90D)
        cols["hurst_short_minus_long"] = short - long
        cols["hurst_change"] = short - short.shift(W30D)

        out = pd.concat({k: v.stack() for k, v in cols.items()}, axis=1)
        out.index = out.index.set_names(["t_obs", "symbol"])
        return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _z(x: pd.DataFrame, w: int, m: int) -> pd.DataFrame:
    sd = x.rolling(w, min_periods=m).std()
    return (x - x.rolling(w, min_periods=m).mean()) / sd.replace(0.0, np.nan)


def _pct(x: pd.DataFrame, w: int, m: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=m).rank(pct=True)


def _hurst(cum: pd.DataFrame, w: int, m: int) -> pd.DataFrame:
    """Hurst exponent from mean squared displacement across a lag ladder.

    MSD(k) = E[(X_t - X_{t-k})^2] scales as k^(2H), so H is half the slope of
    log MSD against log k. Computed as a fixed linear combination of rolling
    means -- one pass per lag rather than a regression per bar.
    """
    logs = np.log(np.array(HURST_LAGS, dtype="float64"))
    lx = logs - logs.mean()
    denom = float((lx ** 2).sum())
    acc = None
    for lag, w_i in zip(HURST_LAGS, lx):
        d2 = (cum - cum.shift(lag)) ** 2
        msd = d2.rolling(w, min_periods=m).mean()
        term = np.log(msd.where(msd > 0)) * w_i
        acc = term if acc is None else acc + term
    return acc / (2.0 * denom)
