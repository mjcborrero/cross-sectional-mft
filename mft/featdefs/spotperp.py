"""Family G -- Spot vs perp, basis and premium. Per FEATURE_LIST_FROZEN.md §3.

Two continuously traded venues for the same asset, with materially different
participant mixes. This is unavailable in other asset classes and was bet #2 in
FEATURE_EXPLORATION.md §13 -- omitted from the original freeze by oversight and
added by Amendment 1.

G11 IS A CONTEXT COLUMN, NOT A PER-COIN FEATURE. Binance lists quarterly
delivery futures for BTC and ETH only (`DELIVERY_UNDERLYINGS_UM` in
mft/data/specs.py), so a per-coin term-basis slope is unavailable for 18 of 20
coins and would fail per-coin coverage by construction. What it actually
measures is a MARKET-WIDE term structure, so it is computed from BTC's front
contract and declared bar-level -- usable through interactions, per §12, rather
than deleted for being what it is.

G6 uses a lead-lag DIFFERENCE rather than a full cross-correlation profile:
corr(perp_t, spot_{t-1}) - corr(perp_{t-1}, spot_t) over 7 days of 1m returns.
Positive means the perp moves first. Two rolling correlations per coin instead
of a profile at every lag, which keeps an already kline-heavy family tractable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.featbuild import MS_HOUR, MS_MIN, Family, GridContext
from mft.featdefs.liquidity import _Win, _asof_last

W8 = 8 * MS_HOUR
W7D_MIN = 7 * 24 * 60
W30D_BARS = 90
MIN30D = 54

PERP_COLS = ["close_time", "close", "quote_volume", "taker_buy_quote_volume",
             "trade_count"]
SPOT_COLS = PERP_COLS
PREM_COLS = ["close_time", "close", "high", "low"]


class SpotPerp(Family):
    code = "G"
    name = "Spot vs perp, basis and premium"
    # Transient BY CONSTRUCTION -- a difference, a ratio to a lagged
    # value, a deviation from own history, or a windowed return sum.
    # Not stable coin characteristics, so Stage 3 persistence does not
    # apply. Classified from construction, not from any score.
    change_columns = (
        "basis_change_8h",
        "perp_spot_return_gap",
    )
    input_datasets = ("perp_klines", "spot_klines", "premium_index")
    context_columns = ("term_basis_slope",)
    warmup_bars = W30D_BARS

    def feature_availability(self, ctx: GridContext) -> dict[str, int]:
        """Quarterly delivery futures begin later than the perp klines.

        Derived from the first contract file on disk, not hardcoded: measured
        at 2021-02, against a grid starting 2020-09. All 415 missing values sit
        in that window and none after it.
        """
        c = ctx._store.contract_symbols("um_delivery_klines", "BTCUSDT")
        if not c:
            return {}
        start = ctx.dataset_start("um_delivery_klines", c[0])
        return {"term_basis_slope": start} if start else {}

    def compute(self, ctx: GridContext) -> pd.DataFrame:
        t_obs = ctx.instants()
        idx = pd.Index(t_obs, name="t_obs")
        symbols = sorted(ctx.grid()["symbol"].unique())
        term = self._term_basis(ctx, t_obs)
        out = {}

        for sym in symbols:
            p = ctx.dataset("perp_klines", sym, PERP_COLS, "close_time")
            s = ctx.dataset("spot_klines", sym, SPOT_COLS, "close_time")
            if p.empty or s.empty:
                continue
            p = p.drop_duplicates("close_time").sort_values("close_time")
            s = s.drop_duplicates("close_time").sort_values("close_time")
            pt = p["close_time"].to_numpy(dtype=np.int64)
            st = s["close_time"].to_numpy(dtype=np.int64)

            pc = p["close"].to_numpy("float64")
            pqv = p["quote_volume"].to_numpy("float64")
            ptb = p["taker_buy_quote_volume"].to_numpy("float64")
            pnt = p["trade_count"].to_numpy("float64")

            wp = _Win(pt, t_obs, W8)
            ws = _Win(st, t_obs, W8)

            # ---- basis, on a common 1m clock ------------------------------
            spot_at_p = _asof_last(st, s["close"].to_numpy("float64"), pt)
            basis_1m = pc / np.where(spot_at_p > 0, spot_at_p, np.nan) - 1.0
            basis = pd.Series(_asof_last(pt, basis_1m, t_obs), index=idx)
            g1 = _z(basis)
            g2 = basis - basis.shift(1)

            # ---- volume and flow contrasts --------------------------------
            pv, sv = wp.sum(pqv), ws.sum(s["quote_volume"].to_numpy("float64"))
            g3 = _z(pd.Series(np.log(np.where((pv > 0) & (sv > 0), pv / sv, np.nan)),
                              index=idx))

            p_share = wp.sum(ptb) / np.where(pv > 0, pv, np.nan)
            s_share = ws.sum(s["taker_buy_quote_volume"].to_numpy("float64")) / \
                np.where(sv > 0, sv, np.nan)
            g4 = pd.Series(p_share - s_share, index=idx)

            p_ret = _asof_last(pt, pc, t_obs) / _asof_last(pt, pc, t_obs - W8) - 1.0
            sc = s["close"].to_numpy("float64")
            s_ret = _asof_last(st, sc, t_obs) / _asof_last(st, sc, t_obs - W8) - 1.0
            g5 = pd.Series(p_ret - s_ret, index=idx)

            g7 = _z(pd.Series(
                np.log(np.where((wp.sum(pnt) > 0) & (ws.sum(s["trade_count"]
                                                            .to_numpy("float64")) > 0),
                                wp.sum(pnt) / ws.sum(s["trade_count"]
                                                     .to_numpy("float64")), np.nan)),
                index=idx))

            # ---- G6: which venue moves first ------------------------------
            g6 = self._leadlag(pt, pc, st, sc, t_obs)

            df = pd.DataFrame({
                "basis_z": g1, "basis_change_8h": g2,
                "perp_spot_vol_ratio": g3, "aggressor_divergence": g4,
                "perp_spot_return_gap": g5, "perp_spot_leadlag": g6,
                "perp_spot_tradecount_ratio": g7,
            }, index=idx)

            # ---- premium index block --------------------------------------
            pi = ctx.dataset("premium_index", sym, PREM_COLS, "close_time")
            if pi.empty:
                df["premium_vol"] = np.nan
                df["premium_time_above_zero"] = np.nan
                df["premium_range"] = np.nan
            else:
                pi = pi.drop_duplicates("close_time").sort_values("close_time")
                it = pi["close_time"].to_numpy(dtype=np.int64)
                iv = pi["close"].to_numpy("float64")
                wi = _Win(it, t_obs, W8)
                nn = wi.count_valid(np.isfinite(iv))
                m1 = wi.sum(np.nan_to_num(iv)) / np.where(nn > 0, nn, np.nan)
                m2 = wi.sum(np.nan_to_num(iv) ** 2) / np.where(nn > 0, nn, np.nan)
                df["premium_vol"] = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
                df["premium_time_above_zero"] = \
                    wi.sum((iv > 0).astype("float64")) / np.where(nn > 0, nn, np.nan)
                rng = _win_range(it, pi["high"].to_numpy("float64"),
                                 pi["low"].to_numpy("float64"), t_obs)
                df["premium_range"] = rng / np.where(np.abs(m1) > 1e-9,
                                                     np.abs(m1), np.nan)

            df["term_basis_slope"] = term
            out[sym] = df

        res = pd.concat(out, names=["symbol"]).reorder_levels(["t_obs", "symbol"])
        return res.replace([np.inf, -np.inf], np.nan).sort_index()

    @staticmethod
    def _leadlag(pt, pc, st, sc, t_obs) -> pd.Series:
        """corr(perp_t, spot_{t-1}) - corr(perp_{t-1}, spot_t), 7d of 1m."""
        grid = np.arange(pt.min(), pt.max() + MS_MIN, MS_MIN, dtype=np.int64)
        pp = pd.Series(_asof_last(pt, pc, grid), index=grid).pct_change()
        ss = pd.Series(_asof_last(st, sc, grid), index=grid).pct_change()
        fwd = pp.rolling(W7D_MIN, min_periods=W7D_MIN // 3).corr(ss.shift(1))
        bwd = pp.shift(1).rolling(W7D_MIN, min_periods=W7D_MIN // 3).corr(ss)
        d = (fwd - bwd).to_numpy("float64")
        return pd.Series(_asof_last(grid, d, t_obs),
                         index=pd.Index(t_obs, name="t_obs"))

    @staticmethod
    def _term_basis(ctx: GridContext, t_obs: np.ndarray) -> pd.Series:
        """BTC front-quarterly vs perp, annualised. Market-wide, so bar-level."""
        idx = pd.Index(t_obs, name="t_obs")
        try:
            contracts = ctx._store.contract_symbols("um_delivery_klines", "BTCUSDT")
        except Exception:
            return pd.Series(np.nan, index=idx)
        if not contracts:
            return pd.Series(np.nan, index=idx)

        perp = ctx.dataset("perp_klines", "BTCUSDT", ["close_time", "close"],
                           "close_time")
        if perp.empty:
            return pd.Series(np.nan, index=idx)
        ppt = perp["close_time"].to_numpy(dtype=np.int64)
        ppc = perp["close"].to_numpy("float64")
        perp_at = _asof_last(ppt, ppc, t_obs)

        best = np.full(len(t_obs), np.nan)
        best_ttm = np.full(len(t_obs), np.inf)
        for c in contracts:
            exp = pd.Timestamp(f"20{c[-6:-4]}-{c[-4:-2]}-{c[-2:]}", tz="UTC")
            exp_ms = int(exp.value // 10 ** 6)
            d = ctx.dataset("um_delivery_klines", c, ["close_time", "close"],
                            "close_time")
            if d.empty:
                continue
            dt = d["close_time"].to_numpy(dtype=np.int64)
            dv = _asof_last(dt, d["close"].to_numpy("float64"), t_obs)
            ttm = (exp_ms - t_obs) / (365.0 * 24 * MS_HOUR)
            live = np.isfinite(dv) & (ttm > 1.0 / 365.0) & (ttm < best_ttm)
            ann = (dv / np.where(perp_at > 0, perp_at, np.nan) - 1.0) / \
                np.where(ttm > 0, ttm, np.nan)
            best = np.where(live, ann, best)
            best_ttm = np.where(live, ttm, best_ttm)
        return pd.Series(best, index=idx)


def _win_range(ct, hi, lo, t_obs) -> np.ndarray:
    s_hi = pd.Series(hi, index=ct).rolling(480, min_periods=120).max()
    s_lo = pd.Series(lo, index=ct).rolling(480, min_periods=120).min()
    return _asof_last(ct, (s_hi - s_lo).to_numpy("float64"), t_obs)


def _z(s: pd.Series) -> pd.Series:
    sd = s.rolling(W30D_BARS, min_periods=MIN30D).std()
    return (s - s.rolling(W30D_BARS, min_periods=MIN30D).mean()) / sd.replace(0.0, np.nan)
