"""Family I -- inherited library, the part Family H did not build.

WHY THIS EXISTS

Family H rebuilt only `anatomy`, `price_ladder` and `hurst`, arguing the other
modules' mechanisms were "already covered by designed families". That was an
ASSERTION of coverage, not a measurement -- and Stage 4 exists precisely to
test coverage claims empirically, which it cannot do for features that were
never built. An audit found 60 of the 65 admitted base features missing.

Sorting those 60 honestly gave three groups:

  CORRECTLY ABSENT (~12). bar_duration_* are dollar-bar durations and are
  CONSTANT on a fixed 8h clock -- the clock audit admitted anatomy.py wholesale
  with a caveat only about lags, which was an error: duration is more
  fundamentally clock-bound than the lags were. ls_* are the long/short ratios
  already removed as B3-B5 for the 2022 publication hole. hours_since/to_funding
  are degenerate on a funding-aligned grid.

  GENUINELY COVERED (~29). Monotone rescalings that cannot reorder a
  cross-section: funding_rate_annualized, basis, basis_bps. And exact
  duplicates under another name: basis_zscore_30d IS basis_z,
  bar_close_minus_vwap IS close_vs_vwap.

  GENUINELY MISSING (19). Built here, of which 16 survive: two were
  monotone maps of a sibling or of a Family H column (rank correlation
  1.0000000000 in 100% of bars, measured not assumed) and are removed
  at construction rather than left for Stage 4 to absorb, since a
  column with zero ordering content should not spend budget.

Names follow the originals for traceability. Where a window had to be
reinterpreted for the 8h clock that is stated, not silently changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.featbuild import MS_HOUR, MS_MIN, Family, GridContext
from mft.paths import DATA_DIR
from mft.featdefs.liquidity import _Win, _asof_last
from mft.rolling import rolling_moments

W8 = 8 * MS_HOUR
W30D = 90          # 30 days of 8h steps
MIN30D = 54
STALE_MS = 2 * MS_HOUR

K_COLS = ["close_time", "close", "high", "low"]


class Inherited2(Family):
    code = "I"
    name = "Inherited library, part 2 (the mechanisms Family H skipped)"
    input_datasets = ("perp_klines",)
    # term_basis.py's slope/days-to-expiry are NOT rebuilt here: Family G
    # already emits `term_basis_slope` as a context column from the same
    # delivery-futures curve. Two names for one mechanism is exactly the
    # duplication Stage 4 exists to delete, so it is not created.
    change_columns = (
        "basis_acceleration", "funding_rate_change", "funding_rate_lag_1",
        "funding_rate_lag_2", "beta_change_3d_7d", "bar_range_zscore_30d",
        "bpv_zscore_30d", "max_daily_ret_30d",
    )
    warmup_bars = W30D

    def compute(self, ctx: GridContext) -> pd.DataFrame:
        grid = ctx.grid()
        t_obs = ctx.instants()
        idx = pd.Index(t_obs, name="t_obs")
        symbols = sorted(grid["symbol"].unique())

        P = grid.pivot(index="t_obs", columns="symbol", values="px_obs").reindex(idx)

        # Shared residual machinery, same construction as Families E and H.
        bi = ctx.artifact("beta_idiovol_8h")
        BETA = bi.pivot(index="t_obs", columns="symbol", values="beta").reindex(idx)
        SIG = bi.pivot(index="t_obs", columns="symbol", values="sigma_eps").reindex(idx)
        wts = ctx.artifact("market_weights_30d")
        W = wts.pivot(index="t_obs", columns="symbol", values="weight") \
               .reindex(idx).reindex(columns=P.columns)
        R = P / P.shift(1) - 1.0
        M = W.shift(1).where(R.notna())
        M = M.div(M.sum(axis=1), axis=0)
        rm = (M * R).sum(axis=1, min_count=1).where(M.notna().sum(axis=1) >= 10)
        E = R.sub(BETA.mul(rm, axis=0)).div(SIG)

        # ---- skew.py: downside vs upside asymmetry -----------------------
        neg = E.where(E < 0, 0.0) ** 2
        pos = E.where(E > 0, 0.0) ** 2
        dn = neg.rolling(W30D, min_periods=MIN30D).mean()
        up = pos.rolling(W30D, min_periods=MIN30D).mean()
        semivar_ratio = dn / up.replace(0.0, np.nan)
        r24 = P / P.shift(3) - 1.0
        max_daily = r24.rolling(W30D, min_periods=MIN30D).max()

        # ---- price_ladder.py: vol across SAMPLING rates ------------------
        h = ctx.artifact("returns_1h", ts_col="t")
        H1 = h.pivot(index="t", columns="symbol", values="ret")
        hg = np.asarray(H1.index, dtype=np.int64)
        v1h = H1.rolling(30 * 24, min_periods=int(30 * 24 * 0.6)).std() * np.sqrt(8)
        v1h_at = pd.DataFrame(
            {c: _asof_last(hg, v1h[c].to_numpy("float64"), t_obs) for c in v1h.columns},
            index=idx).reindex(columns=P.columns)
        v24 = (E.rolling(3, min_periods=2).sum()
               .rolling(W30D, min_periods=MIN30D).std() / np.sqrt(3.0))
        # Only the RATIO is emitted. A `log` of it was built first and measured
        # at rank correlation 1.0000000000 with it in 100% of bars -- `log` is
        # monotone, so it cannot reorder a cross-section and carries zero
        # information under a ranking objective. Same finding, and same reason,
        # as the relative transforms in FEATURE_LIST_FROZEN.md §4.2.
        vol_ratio_1h_24h = v1h_at.div(SIG) / v24.replace(0.0, np.nan)

        # ---- co_movement.py: beta at two horizons ------------------------
        x1 = h.drop_duplicates("t").set_index("t")["r_m"].reindex(H1.index)
        b3 = rolling_moments(H1, x1, 3 * 24, int(3 * 24 * 0.6)).beta_ols
        b7 = rolling_moments(H1, x1, 7 * 24, int(7 * 24 * 0.6)).beta_ols
        d37 = (b3 - b7)
        beta_change = pd.DataFrame(
            {c: _asof_last(hg, d37[c].to_numpy("float64"), t_obs) for c in d37.columns},
            index=idx).reindex(columns=P.columns)

        # ---- per-symbol kline and metric work ----------------------------
        rng, closeloc, bpv, basis_a, basis_abs = {}, {}, {}, {}, {}
        oi_dd, oi_dl, f_chg, f_l1, f_l2 = {}, {}, {}, {}, {}
        for sym in symbols:
            k = ctx.dataset("perp_klines", sym, K_COLS, "close_time")
            if k.empty:
                continue
            k = k.drop_duplicates("close_time").sort_values("close_time")
            kt = k["close_time"].to_numpy(dtype=np.int64)
            kc = k["close"].to_numpy("float64")

            hi = pd.Series(k["high"].to_numpy("float64"), index=kt) \
                .rolling(480, min_periods=120).max()
            lo = pd.Series(k["low"].to_numpy("float64"), index=kt) \
                .rolling(480, min_periods=120).min()
            H_ = _asof_last(kt, hi.to_numpy("float64"), t_obs)
            L_ = _asof_last(kt, lo.to_numpy("float64"), t_obs)
            C_ = _asof_last(kt, kc, t_obs)
            span = H_ - L_
            rng[sym] = np.where(C_ > 0, span / C_, np.nan)
            closeloc[sym] = np.where(span > 0, (C_ - L_) / span, np.nan)

            # bipower vs realised variance, 8h window of 1m returns
            r1 = np.diff(np.log(np.where(kc > 0, kc, np.nan)), prepend=np.nan)
            w = _Win(kt, t_obs, W8)
            rv = w.sum(np.nan_to_num(r1 ** 2))
            ar = np.abs(np.nan_to_num(r1))
            bp = (np.pi / 2.0) * w.sum(ar * np.roll(ar, 1))
            bpv[sym] = np.where(rv > 0, bp / rv, np.nan)

            # basis, from spot; second difference is the acceleration
            s = ctx.dataset("spot_klines", sym, ["close_time", "close"], "close_time")
            if s.empty:
                basis_a[sym] = np.full(len(t_obs), np.nan)
                basis_abs[sym] = np.full(len(t_obs), np.nan)
            else:
                s = s.drop_duplicates("close_time").sort_values("close_time")
                sp = _asof_last(s["close_time"].to_numpy(dtype=np.int64),
                                s["close"].to_numpy("float64"), t_obs)
                b = np.where(sp > 0, C_ / sp - 1.0, np.nan)
                bs = pd.Series(b, index=idx)
                basis_abs[sym] = np.abs(b)
                basis_a[sym] = (bs - 2 * bs.shift(1) + bs.shift(2)).to_numpy()

            m = ctx.dataset("metrics", sym, ["create_time", "sum_open_interest"],
                            "create_time")
            if m.empty:
                oi_dd[sym] = np.full(len(t_obs), np.nan)
                oi_dl[sym] = np.full(len(t_obs), np.nan)
            else:
                m = m.drop_duplicates("create_time").sort_values("create_time")
                oi = pd.Series(_asof_last(m["create_time"].to_numpy(dtype=np.int64),
                                          m["sum_open_interest"].to_numpy("float64"),
                                          t_obs), index=idx)
                oi_dd[sym] = (oi / oi.rolling(W30D, min_periods=MIN30D).max() - 1.0).to_numpy()
                oi_dl[sym] = (oi / oi.rolling(W30D, min_periods=MIN30D).min() - 1.0).to_numpy()

            f = ctx.dataset("funding", sym, ["calc_time", "last_funding_rate"],
                            "calc_time")
            if f.empty:
                f_chg[sym] = f_l1[sym] = f_l2[sym] = np.full(len(t_obs), np.nan)
            else:
                f = f.drop_duplicates("calc_time").sort_values("calc_time")
                ft = f["calc_time"].to_numpy(dtype=np.int64)
                fv = f["last_funding_rate"].to_numpy("float64")
                # LAGS ARE TAKEN IN SETTLEMENT SPACE, NOT ON THE GRID.
                #
                # Funding settles every 8h at the same instants the grid uses,
                # with a few ms of jitter: measured on BTCUSDT, 3,078 of 5,481
                # settlements land exactly on the boundary and the rest 1-8 ms
                # after it. Against a grid stamped at the boundary that jitter
                # decides which bar a settlement falls in -- 855 bars capture
                # two settlements and 854 capture none. A grid-space
                # `cur - cur.shift(1)` is therefore exactly 0 for EVERY coin at
                # once in 18.5% of bars (measured: 864 bars, all-exactly-zero),
                # which is a timestamp artifact, not funding standing still.
                #
                # Indexing into the settlement sequence removes the artifact
                # entirely: j is the position of the last settlement at or
                # before t_obs, so j-1 is unambiguously the previous one however
                # the boundary falls. Still strictly point-in-time -- j depends
                # on nothing after t_obs.
                j = np.searchsorted(ft, t_obs, side="right") - 1
                f_chg[sym] = _seq(fv, j, 0) - _seq(fv, j, 1)
                f_l1[sym] = _seq(fv, j, 1)
                f_l2[sym] = _seq(fv, j, 2)

        def frame(d):
            return pd.DataFrame(d, index=idx).reindex(columns=P.columns)

        range_frame = frame(rng)
        cols = {
            "max_daily_ret_30d": max_daily,
            "realized_vol_ratio_1h_24h": vol_ratio_1h_24h,
            "beta_change_3d_7d": beta_change,
            "bar_range_zscore_30d": _z(range_frame),
            "bar_close_location": frame(closeloc),
            "bpv_zscore_30d": _z(frame(bpv)),
            "basis_abs": frame(basis_abs),
            "basis_acceleration": frame(basis_a),
            "funding_rate_change": frame(f_chg),
            "funding_rate_lag_1": frame(f_l1),
            "funding_rate_lag_2": frame(f_l2),
        }
        out = pd.concat({k: v.stack() for k, v in cols.items()}, axis=1)
        out.index = out.index.set_names(["t_obs", "symbol"])
        return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _seq(v: np.ndarray, j: np.ndarray, back: int) -> np.ndarray:
    """v[j - back], NaN where that position does not exist."""
    k = j - back
    ok = k >= 0
    out = np.full(len(j), np.nan)
    out[ok] = v[k[ok]]
    return out


def _z(x: pd.DataFrame) -> pd.DataFrame:
    sd = x.rolling(W30D, min_periods=MIN30D).std()
    return (x - x.rolling(W30D, min_periods=MIN30D).mean()) / sd.replace(0.0, np.nan)
