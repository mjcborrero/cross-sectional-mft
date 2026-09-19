"""Family B -- Positioning. Per docs/FEATURE_LIST_FROZEN.md §3.

Reads `metrics` (5-minute open interest and four long/short ratios), which is a
different alignment problem from Family A: funding settles once per grid step,
whereas metrics arrive ~96 times per step. The last observation at or before
t_obs is taken, with a staleness rejection so a data gap is not carried forward
as a live reading.

B3, B4 and B5 -- the three long/short-ratio contrasts -- were REMOVED at Stage
0 on 2026-08-10. Binance stops publishing `sum_toptrader_long_short_ratio` and
`sum_taker_long_short_vol_ratio` for essentially all of 2022 (~99% null across
Q1-Q4), which is a hole INSIDE the window where the dataset otherwise exists.
Corrected coverage was 68.6%, against a 95% bar.

That is a data limitation, not a rejected hypothesis. The mechanism was never
tested. If Binance ever backfills those fields the features are worth
restoring -- the contrasts between position-weighted, account-weighted and
flow-weighted ratios are genuinely different quantities.

What survives reads only `sum_open_interest` and `sum_open_interest_value`,
which have zero nulls across the whole dataset span.

POINT-IN-TIME. Everything uses `create_time <= t_obs`, enforced by the
GridContext on load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.featbuild import MS_HOUR, MS_MIN, Family, GridContext

METRIC_COLS = ["create_time", "sum_open_interest", "sum_open_interest_value"]

STALE_MS = 2 * MS_HOUR      # a 5-minute series silent for 2h is a gap
W30_H = 30 * 24             # 30 days of hourly observations
MIN30_H = int(W30_H * 0.6)


def _asof(src_t: np.ndarray, vals: np.ndarray, at: np.ndarray,
          stale_ms: int) -> np.ndarray:
    """Last value at or before each instant; NaN if staler than `stale_ms`."""
    out = np.full(len(at), np.nan)
    if len(src_t) == 0:
        return out
    pos = np.searchsorted(src_t, at, side="right") - 1
    ok = pos >= 0
    if ok.any():
        p = pos[ok]
        fresh = (at[ok] - src_t[p]) <= stale_ms
        sel = np.where(ok)[0][fresh]
        out[sel] = vals[p][fresh]
    return out


class Positioning(Family):
    code = "B"
    name = "Positioning (open interest and long/short ratios)"
    # Transient BY CONSTRUCTION -- a difference, a ratio to a lagged
    # value, a deviation from own history, or a windowed return sum.
    # Not stable coin characteristics, so Stage 3 persistence does not
    # apply. Classified from construction, not from any score.
    change_columns = (
        "oi_velocity_24h",
    )
    input_datasets = ("metrics",)

    def compute(self, ctx: GridContext) -> pd.DataFrame:
        grid = ctx.grid()
        t_obs = ctx.instants()
        symbols = sorted(grid["symbol"].unique())

        # Hourly returns, for the OI-vs-price correlation. Clipping this
        # artifact is truncation-safe: each row depends on prices at t-1 and t.
        h = ctx.artifact("returns_1h", ts_col="t")
        H1 = h.pivot(index="t", columns="symbol", values="ret")
        hgrid = np.asarray(H1.index, dtype=np.int64)

        qv = grid.pivot(index="t_obs", columns="symbol", values="qv_window") \
                 .reindex(t_obs)

        # Trade count summed over each 8h window, for B6.
        tc = self._trade_counts(ctx, symbols, t_obs)

        idx = pd.Index(t_obs, name="t_obs")
        out = {}
        for sym in symbols:
            m = ctx.dataset("metrics", sym, METRIC_COLS, "create_time")
            if m.empty:
                continue
            m = m.drop_duplicates("create_time").sort_values("create_time")
            mt = m["create_time"].to_numpy(dtype=np.int64)

            def at_grid(col: str, at: np.ndarray = t_obs) -> pd.Series:
                return pd.Series(
                    _asof(mt, m[col].to_numpy(dtype="float64"), at, STALE_MS),
                    index=pd.Index(at, name="t_obs"))

            oi_val = at_grid("sum_open_interest_value")
            oi_ct = at_grid("sum_open_interest")

            # B1: 24h OI velocity -- 3 grid steps back, scale-free.
            oi_prev = oi_ct.shift(3)
            b1 = (oi_ct - oi_prev) / oi_prev.replace(0.0, np.nan)

            # B2: rolling corr(dOI, dprice) on the HOURLY grid, 30d.
            oi_h = pd.Series(_asof(mt, m["sum_open_interest"].to_numpy("float64"),
                                   hgrid, STALE_MS), index=H1.index)
            d_oi = oi_h.pct_change().replace([np.inf, -np.inf], np.nan)
            d_px = H1[sym] if sym in H1.columns else pd.Series(np.nan, index=H1.index)
            b2_h = d_oi.rolling(W30_H, min_periods=MIN30_H).corr(d_px)
            b2 = pd.Series(_asof(hgrid, b2_h.to_numpy("float64"), t_obs, STALE_MS),
                           index=idx)

            # B6: OI change per trade -- many small builders vs few large.
            n_tr = tc[sym] if sym in tc.columns else pd.Series(np.nan, index=idx)
            b6 = (oi_ct - oi_ct.shift(1)) / n_tr.replace(0.0, np.nan)

            # B7: turnover -- both legs in USD, so the ratio is scale-free.
            q = qv[sym] if sym in qv.columns else pd.Series(np.nan, index=idx)
            b7 = q / oi_val.replace(0.0, np.nan)

            out[sym] = pd.DataFrame({
                "oi_velocity_24h": b1,
                "oi_price_corr": b2,
                "oi_per_trade": b6,
                "turnover": b7,
            }, index=idx)

        df = pd.concat(out, names=["symbol"]).reorder_levels(["t_obs", "symbol"])
        return df.replace([np.inf, -np.inf], np.nan).sort_index()

    @staticmethod
    def _trade_counts(ctx: GridContext, symbols: list[str],
                      t_obs: np.ndarray) -> pd.DataFrame:
        """Trade count summed over each trailing 8h window, from 1m klines."""
        step = 8 * MS_HOUR
        cols = {}
        for sym in symbols:
            k = ctx.dataset("perp_klines", sym, ["close_time", "trade_count"],
                            "close_time")
            if k.empty:
                continue
            ct = k["close_time"].to_numpy(dtype=np.int64)
            cum = np.concatenate(([0.0], np.cumsum(k["trade_count"].to_numpy("float64"))))
            hi = np.searchsorted(ct, t_obs, side="right")
            lo = np.searchsorted(ct, t_obs - step, side="right")
            v = cum[hi] - cum[lo]
            cols[sym] = np.where(hi > lo, v, np.nan)
        return pd.DataFrame(cols, index=pd.Index(t_obs, name="t_obs"))
