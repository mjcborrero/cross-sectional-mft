"""Family C -- Liquidity and microstructure. Per docs/FEATURE_LIST_FROZEN.md §3.

Everything here is estimated from 1-minute klines and the 1-minute premium
index. No order book, no trade prints -- aggTrades is not downloaded, so the
spread and impact measures are the *free proxies* Roll (1984) and Corwin-
Schultz (2012) were designed to be.

All window statistics are computed from cumulative sums and `searchsorted`,
so a rolling regression over 480 or 1440 one-minute observations costs one
pass rather than one fit per bar.

THREE CONSTRUCTION CHOICES, recorded because they are judgement, not spec:

  Roll's estimator is undefined when the serial covariance is positive, which
  happens often at 1m in trending markets. The standard treatment since Roll
  himself is to floor the implied spread at zero rather than discard the
  observation, and that is what is done here. Reported, not hidden: the share
  of floored observations is printed at build time.

  C6 is named `vpin_perp_z` after the frozen list, but true VPIN buckets by
  equal VOLUME and needs trade-level data. What is computed here is the bulk
  order-flow imbalance over equal-TIME (1m) aggregation, which is what klines
  support. It is a proxy for the same quantity, and calling it VPIN without
  this note would overstate it.

  Windows are 8h for within-bar quantities (matching the decision step) and
  24h for the estimators that need more observations to be stable -- Roll and
  the premium AR(1). Longer windows for noisier estimators, per
  FEATURE_EVALUATION.md Stage 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.featbuild import MS_HOUR, MS_MIN, Family, GridContext

MS_DAY = 24 * MS_HOUR
W8 = 8 * MS_HOUR
W24 = 24 * MS_HOUR
W30D_BARS = 90                    # 30 days of 8h steps
MIN30D = 54

KLINE_COLS = ["close_time", "close", "high", "low", "quote_volume",
              "taker_buy_quote_volume"]


def _cum(v: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.nan_to_num(v, nan=0.0))))


class _Win:
    """Windowed sums over an irregular 1m series, via cumsum + searchsorted."""

    def __init__(self, ct: np.ndarray, t_obs: np.ndarray, span: int):
        self.hi = np.searchsorted(ct, t_obs, side="right")
        self.lo = np.searchsorted(ct, t_obs - span, side="right")
        self.n = (self.hi - self.lo).astype("float64")

    def sum(self, v: np.ndarray) -> np.ndarray:
        c = _cum(v)
        return c[self.hi] - c[self.lo]

    def count_valid(self, mask: np.ndarray) -> np.ndarray:
        return self.sum(mask.astype("float64"))

    def slope(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """cov(x, y) / var(x) over the window -- a rolling OLS slope."""
        ok = np.isfinite(x) & np.isfinite(y)
        xs, ys = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
        n = self.count_valid(ok)
        sx, sy = self.sum(xs), self.sum(ys)
        sxx, sxy = self.sum(xs * xs), self.sum(xs * ys)
        vx = sxx - sx * sx / np.where(n > 0, n, np.nan)
        cxy = sxy - sx * sy / np.where(n > 0, n, np.nan)
        out = np.where((n >= 30) & (vx > 0), cxy / np.where(vx > 0, vx, np.nan), np.nan)
        return out

    def cov(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ok = np.isfinite(x) & np.isfinite(y)
        xs, ys = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
        n = self.count_valid(ok)
        sx, sy, sxy = self.sum(xs), self.sum(ys), self.sum(xs * ys)
        denom = np.where(n > 1, n - 1, np.nan)
        return (sxy - sx * sy / np.where(n > 0, n, np.nan)) / denom


class Liquidity(Family):
    code = "C"
    name = "Liquidity and microstructure"
    input_datasets = ("perp_klines", "premium_index")

    def compute(self, ctx: GridContext) -> pd.DataFrame:
        t_obs = ctx.instants()
        symbols = sorted(ctx.grid()["symbol"].unique())
        idx = pd.Index(t_obs, name="t_obs")
        rolled = {"floored": 0, "roll_total": 0}
        out = {}

        for sym in symbols:
            k = ctx.dataset("perp_klines", sym, KLINE_COLS, "close_time")
            if k.empty:
                continue
            k = k.drop_duplicates("close_time").sort_values("close_time")
            ct = k["close_time"].to_numpy(dtype=np.int64)
            close = k["close"].to_numpy("float64")
            high = k["high"].to_numpy("float64")
            low = k["low"].to_numpy("float64")
            qv = k["quote_volume"].to_numpy("float64")
            tbqv = k["taker_buy_quote_volume"].to_numpy("float64")

            ret = np.concatenate(([np.nan], np.diff(close) / close[:-1]))
            ret_lag = np.concatenate(([np.nan], ret[:-1]))
            signed_qv = 2.0 * tbqv - qv          # net taker buy, in quote units

            w8 = _Win(ct, t_obs, W8)
            w24 = _Win(ct, t_obs, W24)

            # C1 -- minutes with no price change at all.
            flat = np.concatenate(([False], np.diff(close) == 0.0))
            c1 = np.where(w8.n > 0, w8.sum(flat.astype("float64")), np.nan)

            # C2 -- Amihud, per 8h bar: |bar return| / bar dollar volume.
            bar_qv = w8.sum(qv)
            px_now = _asof_last(ct, close, t_obs)
            px_prev = _asof_last(ct, close, t_obs - W8)
            bar_ret = px_now / px_prev - 1.0
            c2_bar = np.abs(bar_ret) / np.where(bar_qv > 0, bar_qv, np.nan)

            # C3 -- Roll's implied spread, 24h of 1m returns.
            cov_rr = w24.cov(ret, ret_lag)
            neg = cov_rr < 0
            rolled["roll_total"] += int(np.isfinite(cov_rr).sum())
            rolled["floored"] += int((np.isfinite(cov_rr) & ~neg).sum())
            c3 = np.where(np.isfinite(cov_rr),
                          np.where(neg, 2.0 * np.sqrt(np.abs(np.where(neg, cov_rr, 0.0))),
                                   0.0), np.nan)

            # C4 -- Kyle's lambda: price response per unit of signed flow.
            c4 = w8.slope(signed_qv, ret)

            # C6 -- bulk order-flow imbalance (VPIN proxy, see module docstring).
            c6_raw = np.abs(w8.sum(signed_qv)) / np.where(bar_qv > 0, bar_qv, np.nan)

            # C7 -- persistence of signed flow across consecutive 8h bars.
            flow8 = pd.Series(w8.sum(signed_qv) / np.where(bar_qv > 0, bar_qv, np.nan),
                              index=idx)
            c7 = flow8.rolling(W30D_BARS, min_periods=MIN30D).corr(flow8.shift(1))

            # C8 -- Corwin-Schultz, from adjacent 8h high-low ranges.
            hi8 = _win_extreme(ct, high, t_obs, W8, np.maximum.reduceat, -np.inf)
            lo8 = _win_extreme(ct, low, t_obs, W8, np.minimum.reduceat, np.inf)
            c8_bar = _corwin_schultz(pd.Series(hi8, index=idx),
                                     pd.Series(lo8, index=idx))

            df = pd.DataFrame({
                "zero_return_minutes": c1,
                "roll_spread": c3,
                "kyle_lambda": c4,
                "vpin_perp_z": _z(pd.Series(c6_raw, index=idx)),
                "corwin_schultz": c8_bar.rolling(W30D_BARS, min_periods=MIN30D).mean(),
            }, index=idx)

            # C5 -- premium AR(1) persistence, 24h of 1m premium.
            pi = ctx.dataset("premium_index", sym, ["close_time", "close"],
                             "close_time")
            if not pi.empty:
                pi = pi.drop_duplicates("close_time").sort_values("close_time")
                pt = pi["close_time"].to_numpy(dtype=np.int64)
                pv = pi["close"].to_numpy("float64")
                pv_lag = np.concatenate(([np.nan], pv[:-1]))
                df["premium_reversion_speed"] = _Win(pt, t_obs, W24).slope(pv_lag, pv)
            else:
                df["premium_reversion_speed"] = np.nan

            out[sym] = df

        if rolled["roll_total"]:
            frac = rolled["floored"] / rolled["roll_total"]
            print(f"  [note] Roll's estimator floored at zero in {frac:.1%} of "
                  f"observations (positive serial covariance -- see module docstring)")

        res = pd.concat(out, names=["symbol"]).reorder_levels(["t_obs", "symbol"])
        return res.replace([np.inf, -np.inf], np.nan).sort_index()


def _asof_last(ct: np.ndarray, v: np.ndarray, at: np.ndarray) -> np.ndarray:
    i = np.searchsorted(ct, at, side="right") - 1
    out = np.full(len(at), np.nan)
    ok = i >= 0
    out[ok] = v[i[ok]]
    return out


def _win_extreme(ct, v, t_obs, span, _unused, fill) -> np.ndarray:
    """Max/min of v over (t-span, t] -- via pandas rolling on a 1m index."""
    s = pd.Series(v, index=pd.Index(ct, name="t"))
    n = max(1, span // MS_MIN)
    agg = s.rolling(n, min_periods=max(2, n // 4))
    r = agg.max() if fill == -np.inf else agg.min()
    return _asof_last(ct, r.to_numpy("float64"), t_obs)


def _corwin_schultz(hi: pd.Series, lo: pd.Series) -> pd.Series:
    """Corwin-Schultz (2012) spread from two adjacent high-low ranges."""
    h2 = np.log(hi / lo) ** 2
    beta = h2 + h2.shift(1)
    hi2 = np.maximum(hi, hi.shift(1))
    lo2 = np.minimum(lo, lo.shift(1))
    gamma = np.log(hi2 / lo2) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return s.clip(lower=0.0)          # negative estimates -> 0, per the paper


def _z(s: pd.Series) -> pd.Series:
    sd = s.rolling(W30D_BARS, min_periods=MIN30D).std()
    return (s - s.rolling(W30D_BARS, min_periods=MIN30D).mean()) / sd.replace(0.0, np.nan)
