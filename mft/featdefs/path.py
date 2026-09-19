"""Family D -- Path and attention. Per docs/FEATURE_LIST_FROZEN.md §3.

Every decision collapses 480 one-minute bars into a single number, and the
path is normally thrown away. This family keeps it: *how* a move happened,
not only how big it was.

All eleven come from 1m perp klines. Window statistics reuse the cumsum +
searchsorted machinery from Family C rather than duplicating it -- the helpers
are imported from `liquidity` deliberately, since a second copy would drift.

TWO CONSTRUCTION CHOICES worth naming:

  D5 `jump_share` is NOT floored at zero. Barndorff-Nielsen-Shephard's relative
  jump measure (RV-BV)/RV goes negative when bipower exceeds realised variance,
  which is noise rather than signal -- but flooring it would build the same 40%
  tie group that limits `roll_spread` in Family C. A negative reading simply
  means no jump was detected, and keeping it continuous preserves ranking
  resolution.

  D7 `move_timing` normalises by total absolute movement rather than by the net
  return. Dividing by the net return explodes whenever a bar round-trips to
  near zero, which is exactly the bar where timing is most interesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.featbuild import MS_HOUR, MS_MIN, Family, GridContext
from mft.featdefs.liquidity import _Win, _asof_last

W8 = 8 * MS_HOUR
W1 = 1 * MS_HOUR
W30D_BARS = 90
MIN30D = 54

KLINE_COLS = ["close_time", "close", "high", "low", "quote_volume", "trade_count"]


class PathAttention(Family):
    code = "D"
    name = "Path and attention"
    input_datasets = ("perp_klines",)

    def compute(self, ctx: GridContext) -> pd.DataFrame:
        t_obs = ctx.instants()
        idx = pd.Index(t_obs, name="t_obs")
        out = {}

        for sym in sorted(ctx.grid()["symbol"].unique()):
            k = ctx.dataset("perp_klines", sym, KLINE_COLS, "close_time")
            if k.empty:
                continue
            k = k.drop_duplicates("close_time").sort_values("close_time")
            ct = k["close_time"].to_numpy(dtype=np.int64)
            close = k["close"].to_numpy("float64")
            high = k["high"].to_numpy("float64")
            low = k["low"].to_numpy("float64")
            qv = k["quote_volume"].to_numpy("float64")
            ntr = k["trade_count"].to_numpy("float64")

            r = np.concatenate(([np.nan], np.diff(close) / close[:-1]))
            r_lag = np.concatenate(([np.nan], r[:-1]))
            ok = np.isfinite(r)
            rs = np.where(ok, r, 0.0)

            w = _Win(ct, t_obs, W8)
            n = w.count_valid(ok)
            valid = n >= 60                      # at least 1/8 of the window

            sum_abs = w.sum(np.abs(rs))
            s2 = w.sum(rs ** 2)
            s3 = w.sum(rs ** 3)
            s4 = w.sum(rs ** 4)
            V = w.sum(qv)
            V2 = w.sum(qv ** 2)
            NT = w.sum(ntr)

            px_now = _asof_last(ct, close, t_obs)
            px_0 = _asof_last(ct, close, t_obs - W8)
            net = px_now / px_0 - 1.0

            # D1 -- attention divergence: many small trades vs few large ones.
            lc = pd.Series(np.where(NT > 0, np.log(NT), np.nan), index=idx)
            lv = pd.Series(np.where(V > 0, np.log(V), np.nan), index=idx)
            d1 = (lc - lc.rolling(W30D_BARS, min_periods=MIN30D).mean()) - \
                 (lv - lv.rolling(W30D_BARS, min_periods=MIN30D).mean())

            # D2 -- path efficiency, signed by direction. In [-1, 1].
            d2 = np.where(sum_abs > 0, net / np.where(sum_abs > 0, sum_abs, np.nan),
                          np.nan)

            # D3 -- Herfindahl of the per-minute volumes.
            d3 = np.where(V > 0, V2 / np.where(V > 0, V, np.nan) ** 2, np.nan)

            # D4 -- close relative to the bar's own volume-weighted price.
            vwap = w.sum(close * qv) / np.where(V > 0, V, np.nan)
            d4 = px_now / vwap - 1.0

            # D5 -- relative jump measure, deliberately unfloored (docstring).
            bv = (np.pi / 2.0) * w.sum(np.abs(rs) * np.abs(np.where(np.isfinite(r_lag),
                                                                   r_lag, 0.0)))
            d5 = np.where(s2 > 0, (s2 - bv) / np.where(s2 > 0, s2, np.nan), np.nan)

            # D6 -- signed jump: upside minus downside realised semivariance.
            rs_up = w.sum(np.where(rs > 0, rs ** 2, 0.0))
            rs_dn = w.sum(np.where(rs < 0, rs ** 2, 0.0))
            d6 = np.where(s2 > 0, (rs_up - rs_dn) / np.where(s2 > 0, s2, np.nan),
                          np.nan)

            # D7 -- was the move made early and held, or printed late?
            px_1h_in = _asof_last(ct, close, t_obs - W8 + W1)
            px_1h_end = _asof_last(ct, close, t_obs - W1)
            r_first = px_1h_in / px_0 - 1.0
            r_last = px_now / px_1h_end - 1.0
            d7 = np.where(sum_abs > 0,
                          (r_first - r_last) / np.where(sum_abs > 0, sum_abs, np.nan),
                          np.nan)

            # D8/D9 -- realised skew and kurtosis of the 1m returns.
            d8 = np.where(s2 > 0, np.sqrt(np.maximum(n, 0)) * s3 /
                          np.power(np.where(s2 > 0, s2, np.nan), 1.5), np.nan)
            d9 = np.where(s2 > 0, n * s4 / np.power(np.where(s2 > 0, s2, np.nan), 2),
                          np.nan)

            # D10 -- Parkinson range vol against close-to-close vol.
            hl = np.log(np.where(low > 0, high / low, np.nan)) ** 2
            park = np.sqrt(w.sum(np.nan_to_num(hl)) /
                           (4.0 * np.log(2.0) * np.where(n > 0, n, np.nan)))
            cc = np.sqrt(s2 / np.where(n > 0, n, np.nan))
            d10 = park / np.where(cc > 0, cc, np.nan)

            # D11 -- is volume informative for this coin right now?
            d11 = _win_corr(w, np.abs(rs), qv, ok)

            df = pd.DataFrame({
                "retail_attention_div": d1,
                "path_efficiency": d2,
                "volume_concentration": d3,
                "close_vs_vwap": d4,
                "jump_share": d5,
                "signed_jump": d6,
                "move_timing": d7,
                "realized_skew": d8,
                "realized_kurt": d9,
                "parkinson_ratio": d10,
                "volume_return_coupling": d11,
            }, index=idx)
            out[sym] = df.where(pd.Series(valid, index=idx), np.nan)

        res = pd.concat(out, names=["symbol"]).reorder_levels(["t_obs", "symbol"])
        return res.replace([np.inf, -np.inf], np.nan).sort_index()


def _win_corr(w: _Win, x: np.ndarray, y: np.ndarray,
              ok: np.ndarray) -> np.ndarray:
    """Pearson correlation of x and y inside each window."""
    xs, ys = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    n = w.count_valid(ok)
    sx, sy = w.sum(xs), w.sum(ys)
    sxx, syy, sxy = w.sum(xs * xs), w.sum(ys * ys), w.sum(xs * ys)
    nn = np.where(n > 1, n, np.nan)
    vx = sxx - sx * sx / nn
    vy = syy - sy * sy / nn
    cxy = sxy - sx * sy / nn
    den = np.sqrt(np.where(vx > 0, vx, np.nan) * np.where(vy > 0, vy, np.nan))
    return np.where(n >= 30, cxy / den, np.nan)
