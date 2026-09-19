"""Family E -- Return and regime. Per docs/FEATURE_LIST_FROZEN.md §3.

Built almost entirely from prior artifacts rather than raw klines: the
decision grid, `beta_idiovol_8h`, and `market_weights_30d`. Clipping those is
truncation-safe because every column in them is itself point-in-time -- beta
at t comes from a 30d trailing window, so deleting future data cannot change
it.

A PIT HAZARD THE TRUNCATION TEST WOULD NOT CATCH
------------------------------------------------
Trailing returns here use `px_obs`, NOT `px_fill`.

`px_fill` is the price at `t_obs + lag`. A fill-to-fill trailing return over
[t_fill(t-1), t_fill(t)] therefore closes one hour AFTER the decision instant,
so using it would leak 60 minutes of future information into every feature.

The truncation test cannot see this. It cuts the whole dataset at one date and
checks that earlier rows are unchanged; a bounded per-row lookahead of one
hour survives that check untouched. It has to be prevented by construction,
which is why the market leg is rebuilt here on obs-to-obs returns rather than
reusing `market_index_8h` (which is fill-to-fill by design, correctly, because
it serves the LABEL).

Weights for a trailing window over [t-h, t] are taken at t-h -- known before
the window opens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.featbuild import MS_HOUR, Family, GridContext

STEP = 8 * MS_HOUR
HORIZONS = {"8h": 1, "24h": 3, "72h": 9, "168h": 21}   # in grid steps
W30D = 90        # 30 days of 8h steps
MIN30D = 54
W90D = 270
MIN90D = 162
VR_Q = 9         # 72h / 8h


class ReturnRegime(Family):
    code = "E"
    name = "Return and regime"
    # Transient BY CONSTRUCTION -- a difference, a ratio to a lagged
    # value, a deviation from own history, or a windowed return sum.
    # Not stable coin characteristics, so Stage 3 persistence does not
    # apply. Classified from construction, not from any score.
    change_columns = (
        "beta_momentum",
        "idio_vol_momentum",
        "volume_share_rotation",
        "session_rel_volume",
        "resid_reversal_agreement",
        "resid_reversal_8h",
        "reversal_x_vr",
        "reversal_x_volume",
    )
    warmup_bars = W90D + VR_Q      # the variance ratio is the longest window

    def compute(self, ctx: GridContext) -> pd.DataFrame:
        grid = ctx.grid()
        t_obs = ctx.instants()
        idx = pd.Index(t_obs, name="t_obs")

        P = grid.pivot(index="t_obs", columns="symbol", values="px_obs").reindex(idx)
        QV = grid.pivot(index="t_obs", columns="symbol", values="qv_window").reindex(idx)

        bi = ctx.artifact("beta_idiovol_8h")
        BETA = bi.pivot(index="t_obs", columns="symbol", values="beta").reindex(idx)
        SIG = bi.pivot(index="t_obs", columns="symbol", values="sigma_eps").reindex(idx)
        RATIO = bi.pivot(index="t_obs", columns="symbol", values="ratio").reindex(idx)

        wts = ctx.artifact("market_weights_30d")
        W = wts.pivot(index="t_obs", columns="symbol", values="weight").reindex(idx)
        W = W.reindex(columns=P.columns)

        # ---- residual trailing returns at each horizon --------------------
        resid = {}
        for tag, h in HORIZONS.items():
            R = P / P.shift(h) - 1.0
            wl = W.shift(h)                       # weights known when the window opened
            M = wl.where(R.notna())
            M = M.div(M.sum(axis=1), axis=0)
            rm = (M * R).sum(axis=1, min_count=1).where(M.notna().sum(axis=1) >= 10)
            e = R.sub(BETA.mul(rm, axis=0))
            resid[tag] = e.div(SIG * np.sqrt(h))   # comparable across horizons

        e8 = resid["8h"]

        # E1 -- the benchmark: reversal on the standardised 8h residual.
        e1 = -e8

        # E2 -- multi-horizon reversal, SIGNED MAGNITUDE rather than sign count.
        # The pure sign sum can take only 5 values (it is a sum of four signs),
        # which ties ~4 of 19 coins per bar and fails the resolution gate. The
        # signed-magnitude variant was named in FEATURE_EXPLORATION.md 14.6 as
        # "the natural middle ground and costs the same one slot": it keeps the
        # multi-scale information and the conviction ordering, while remaining
        # continuous. Each leg is already standardised by sigma_eps*sqrt(h), so
        # the legs are directly comparable and a mean is the natural aggregate.
        e2 = -sum(resid[t] for t in HORIZONS) / len(HORIZONS)

        # E3 -- variance ratio on 8h residuals: Var(9-step) / (9 * Var(1-step)).
        e_raw = resid["8h"] * SIG              # back to return units for the ratio
        s9 = e_raw.rolling(VR_Q).sum()
        v9 = s9.rolling(W90D, min_periods=MIN90D).var()
        v1 = e_raw.rolling(W90D, min_periods=MIN90D).var()
        e3 = v9 / (VR_Q * v1.replace(0.0, np.nan))

        # E4/E5 -- interactions built explicitly, not left for a tree to find.
        e4 = e1 * (1.0 - e3)
        vol_surprise = QV / QV.rolling(W30D, min_periods=MIN30D).mean()
        e5 = e1 * (-vol_surprise)

        # E7/E8 -- beta level dynamics, and how stable the estimate itself is.
        e7 = BETA - BETA.shift(W30D)
        e8_inst = BETA.rolling(W30D, min_periods=MIN30D).std()

        # E9 -- idiosyncratic vol expansion.
        e9 = SIG / SIG.shift(W30D) - 1.0

        # E10 -- where the coin sits in its own cycle.
        e10 = P / P.rolling(W30D, min_periods=MIN30D).max() - 1.0

        # E11 -- residual vol term slope, per-hour units at each sampling rate.
        r4 = (P / P.shift(1) - 1.0)            # 8h is the finest grid available
        s_short = resid["24h"].rolling(W30D, min_periods=MIN30D).std() / np.sqrt(3)
        s_long = resid["168h"].rolling(W30D, min_periods=MIN30D).std() / np.sqrt(21)
        e11 = s_short / s_long.replace(0.0, np.nan)

        # E13 -- rotation of attention, measured as volume share drift.
        share = QV.div(QV.sum(axis=1), axis=0)
        e13 = share - share.shift(3)

        # E14 -- volume against this coin's own norm FOR THIS UTC SESSION.
        e14 = _session_relative(QV, t_obs)

        cols = {
            "resid_reversal_8h": e1,
            "resid_reversal_agreement": e2,
            "variance_ratio_72h": e3,
            "reversal_x_vr": e4,
            "reversal_x_volume": e5,
            "idio_share": RATIO,
            "beta_momentum": e7,
            "beta_instability": e8_inst,
            "idio_vol_momentum": e9,
            "drawdown_from_peak": e10,
            "vol_term_slope": e11,
            "volume_share_rotation": e13,
            "session_rel_volume": e14,
        }
        out = pd.concat({k: v.stack() for k, v in cols.items()}, axis=1)
        out.index = out.index.set_names(["t_obs", "symbol"])
        return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _tercile_persistence(x: pd.DataFrame) -> pd.DataFrame:
    """Signed count of consecutive bars in the same cross-sectional tercile.

    Positive in the top tercile, negative in the bottom, zero in the middle --
    so the sign carries direction and the magnitude carries staleness.
    """
    rk = x.rank(axis=1, pct=True)
    top = (rk > 2.0 / 3.0).astype("float64")
    bot = (rk < 1.0 / 3.0).astype("float64")
    state = top - bot
    out = pd.DataFrame(0.0, index=x.index, columns=x.columns)
    run = np.zeros(x.shape[1])
    prev = np.zeros(x.shape[1])
    sv = state.to_numpy()
    for i in range(len(x)):
        cur = sv[i]
        same = (cur == prev) & (cur != 0)
        run = np.where(same, run + 1.0, np.where(cur != 0, 1.0, 0.0))
        out.iloc[i] = run * cur
        prev = cur
    return out.where(x.notna())


def _session_relative(qv: pd.DataFrame, t_obs: np.ndarray) -> pd.DataFrame:
    """log(volume / this coin's own trailing mean for THIS UTC session).

    The 8h grid maps exactly onto Asia / Europe / US sessions, and coins have
    genuinely different session personalities. Comparing a bar against its own
    session's history removes a structure the grid already encodes, rather than
    letting it leak into the ranking.
    """
    hour = pd.to_datetime(t_obs, unit="ms", utc=True).hour
    out = pd.DataFrame(np.nan, index=qv.index, columns=qv.columns)
    lv = np.log(qv.where(qv > 0))
    for h in (0, 8, 16):
        m = hour == h
        sub = lv.loc[m]
        if sub.empty:
            continue
        norm = sub.rolling(30, min_periods=15).mean()
        out.loc[m] = (sub - norm).to_numpy()
    return out
