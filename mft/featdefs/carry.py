"""Family A -- Carry. Five features, per docs/FEATURE_LIST_FROZEN.md §3.

Funding settles every 8h at 00:00 / 08:00 / 16:00 UTC, which is exactly the
decision grid, so each grid step carries exactly one settlement. That alignment
is why this family is the most horizon-matched in the list.

POINT-IN-TIME. Funding stamped at `calc_time <= t_obs` is used. The settlement
AT t_obs is admissible: it is determined at that instant, and the position is
not established until t_obs + 60min, so it is known a full hour before it could
be acted on. The GridContext enforces the cutoff on load, so this family cannot
see beyond it even by mistake.

Demeaning is not cosmetic here. Funding's |rolling mean| / rolling std is
0.904 -- the persistent per-coin level is nearly as large as the variation --
so an undemeaned funding feature would rank coins on a permanent characteristic
rather than on current state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.featbuild import MS_HOUR, Family, GridContext

W30 = 90          # 30 days of 8h settlements
W7 = 21           # 7 days
MIN30 = 54        # 60% of the 30d window
# A3 `funding_clamp_frac` was REMOVED at Stage 1 on 2026-08-07. Two measured
# reasons, neither of which required looking at the target:
#
#   1. The clamp is too rare to rank. Pooled fraction of settlements at the cap
#      is 0.029% -- about 1 in 3,400 -- so over a 30d window it is zero for
#      every coin in ~97% of bars. Measured within-bar variation: 9%.
#   2. The cap is not a constant. SOLUSDT reaches 0.0200 against the 0.0075
#      standard, because Binance raises the cap for volatile pairs. A fixed
#      constant is wrong across the universe.
#
# Its natural continuous form -- "how extreme is funding right now" -- IS
# `funding_z` (A1), so a redefined version would be redundant at Stage 4
# rather than additive. See docs/FEATURE_LIST_FROZEN.md §9.


def _z(s: pd.Series, win: int, minp: int) -> pd.Series:
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - m) / sd.replace(0.0, np.nan)


class Carry(Family):
    code = "A"
    name = "Carry (funding)"
    # Transient BY CONSTRUCTION -- a difference, a ratio to a lagged
    # value, a deviation from own history, or a windowed return sum.
    # Not stable coin characteristics, so Stage 3 persistence does not
    # apply. Classified from construction, not from any score.
    change_columns = (
        "crowd_winning",
        "funding_z",
    )

    def compute(self, ctx: GridContext) -> pd.DataFrame:
        grid = ctx.grid()
        t_obs = ctx.instants()
        symbols = sorted(grid["symbol"].unique())

        # --- trailing 8h return from OBSERVATION prices (known at t_obs) ---
        g = grid.sort_values(["symbol", "t_obs"]).copy()
        gg = g.groupby("symbol", sort=False)
        g["t_prev"] = gg["t_obs"].shift(1)
        g["px_prev"] = gg["px_obs"].shift(1)
        # A gap must never be booked as an 8h return.
        g = g[(g["t_obs"] - g["t_prev"]) == 8 * MS_HOUR]
        g["ret8"] = g["px_obs"] / g["px_prev"] - 1.0
        R = g.pivot(index="t_obs", columns="symbol", values="ret8").reindex(t_obs)

        out = {}
        for sym in symbols:
            f = ctx.dataset("funding", sym, ["calc_time", "last_funding_rate"],
                            "calc_time")
            if f.empty:
                continue
            f = f.drop_duplicates("calc_time").sort_values("calc_time")
            s = f.set_index("calc_time")["last_funding_rate"].astype("float64")

            z_f = _z(s, W30, MIN30)

            cum7 = s.rolling(W7, min_periods=int(W7 * 0.6)).sum()

            sign = np.sign(s)
            # Consecutive same-sign run length, signed. Cumulative-count trick:
            # a new run starts whenever the sign changes.
            grp = (sign != sign.shift(1)).cumsum()
            run = sign.groupby(grp).cumcount() + 1
            persist = run * sign

            r = R[sym] if sym in R.columns else pd.Series(np.nan, index=t_obs)

            # AS-OF onto the grid: the last settlement at or before t_obs.
            # Settlement stamps jitter by seconds, so only ~53% match a grid
            # instant exactly while 100% fall within a minute. Exact-index
            # alignment silently produced a doubled index and destroyed every
            # rolling window; as-of is both correct and point-in-time by
            # construction.
            src = np.asarray(s.index, dtype=np.int64)
            idx = pd.Index(t_obs, name="t_obs")

            def asof(series: pd.Series) -> pd.Series:
                v = series.to_numpy(dtype="float64")
                pos = np.searchsorted(src, t_obs, side="right") - 1
                out_v = np.full(len(t_obs), np.nan)
                ok = pos >= 0
                if ok.any():
                    p = pos[ok]
                    # Reject a stale quote: no settlement within the last 9h
                    # means a data gap, not a carry-forward.
                    fresh = (t_obs[ok] - src[p]) <= 9 * MS_HOUR
                    sel = np.where(ok)[0][fresh]
                    out_v[sel] = v[p][fresh]
                return pd.Series(out_v, index=idx)

            z_r = _z(r, W30, MIN30)
            out[sym] = pd.DataFrame({
                "funding_z": asof(z_f),
                "funding_cum_7d": asof(cum7),
                "funding_sign_persist": asof(persist),
                "crowd_winning": asof(z_f) * z_r.reindex(idx),
            })

        df = pd.concat(out, names=["symbol"]).reorder_levels(["t_obs", "symbol"])
        return df.sort_index()
