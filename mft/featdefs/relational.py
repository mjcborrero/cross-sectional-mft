"""Family F -- Relational. Per docs/FEATURE_LIST_FROZEN.md §3.

Treats the 20 coins as a system rather than 20 independent rows. The target
removes ONE factor; sectors, coupling and propagation live in what is left.

This family spans all three ranking tiers of FEATURE_EXPLORATION.md §12, and
declares which is which rather than letting Stage 1 discover it:

  per-coin   F1 F2 F3 F5 F8 F10 F11   full ranking power
  per-sector F6 F7                    6 sectors -> 6 values per bar, coarse
                                       but genuine within-bar discrimination
  bar-level  F4 F9                    identical for every coin; ZERO standalone
                                       ranking power, usable only via
                                       interactions

THE SECTOR MAP IS FROZEN with the feature list. TRX stays in "Other" despite
being an L1 by technology and a payments/legacy coin by PC2 -- re-assigning
after seeing results would turn the map into a tuning knob with 20 dials.

NETWORK STRUCTURE IS ESTIMATED ON 1h RESIDUALS, never 8h. FEATURE_EXPLORATION
§9.5 measured the pairwise-correlation standard error at 0.105 on 8h returns
against 0.037 on 1h: at 8h, telling a true correlation of 0.3 from 0.4 is
hopeless, and every relational feature rests on exactly that distinction.

LEAVE-ONE-OUT IS MANDATORY. Including self in a peer mean damps the feature by
1/n and does so DIFFERENTLY for each sector size, which would turn sector size
itself into a spurious cross-sectional signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.featbuild import MS_HOUR, Family, GridContext

STEP = 8 * MS_HOUR
HORIZONS = {"8h": 1, "24h": 3, "72h": 9, "168h": 21}
W30D_H = 30 * 24
W90D_H = 90 * 24
MIN90D_H = int(W90D_H * 0.6)
MIN30D_H = int(W30D_H * 0.6)

SECTORS = {
    "BTCUSDT": "majors",   "ETHUSDT": "majors",
    "SOLUSDT": "l1",  "AVAXUSDT": "l1", "NEARUSDT": "l1", "DOTUSDT": "l1",
    "ADAUSDT": "l1",  "SUIUSDT": "l1",  "HBARUSDT": "l1",
    "LTCUSDT": "pow", "BCHUSDT": "pow", "ZECUSDT": "pow",
    "XRPUSDT": "payments", "XLMUSDT": "payments",
    "UNIUSDT": "defi", "AAVEUSDT": "defi", "LINKUSDT": "defi",
    "BNBUSDT": "other", "DOGEUSDT": "other", "TRXUSDT": "other",
}


class Relational(Family):
    code = "F"
    name = "Relational (sector, coupling, propagation)"
    # Transient BY CONSTRUCTION -- a difference, a ratio to a lagged
    # value, a deviation from own history, or a windowed return sum.
    # Not stable coin characteristics, so Stage 3 persistence does not
    # apply. Classified from construction, not from any score.
    change_columns = (
        "sector_decoupling",
        "sector_rel_reversal_8h",
        "sector_rel_agreement",
        "dispersion_contribution",
    )
    sector_columns = ("sector_cohesion", "sector_dispersion")
    context_columns = ("btc_resid_lag", "eth_resid_lag")
    warmup_bars = 270

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

        cols = list(P.columns)
        sec = pd.Series({c: SECTORS.get(c, "other") for c in cols})

        # ---- 8h residuals at each horizon, obs-to-obs (never px_fill) -----
        resid = {}
        for tag, h in HORIZONS.items():
            R = P / P.shift(h) - 1.0
            M = W.shift(h).where(R.notna())
            M = M.div(M.sum(axis=1), axis=0)
            rm = (M * R).sum(axis=1, min_count=1).where(M.notna().sum(axis=1) >= 10)
            resid[tag] = R.sub(BETA.mul(rm, axis=0)).div(SIG * np.sqrt(h))
        e8 = resid["8h"]

        # ---- F1/F2: sector-relative reversal, leave-one-out ---------------
        def sector_rel(e: pd.DataFrame) -> pd.DataFrame:
            out = pd.DataFrame(np.nan, index=e.index, columns=e.columns)
            for s in sec.unique():
                mem = [c for c in cols if sec[c] == s]
                if len(mem) < 2:
                    continue
                blk = e[mem]
                tot = blk.sum(axis=1, min_count=1)
                cnt = blk.notna().sum(axis=1)
                for c in mem:                       # leave-one-out peer mean
                    peers = (tot - blk[c].fillna(0.0)) / (cnt - blk[c].notna()).replace(0, np.nan)
                    out[c] = blk[c] - peers
            return out

        f1 = -sector_rel(e8)
        # Signed magnitude, not a sign count: a sum of four signs takes only 5
        # values and fails the resolution gate (E2, same construction).
        f2 = -sum(sector_rel(resid[t]) for t in HORIZONS) / len(HORIZONS)

        # ---- 1h residuals, for every correlation below --------------------
        h1 = ctx.artifact("returns_1h", ts_col="t")
        R1 = h1.pivot(index="t", columns="symbol", values="ret")
        rm1 = h1.drop_duplicates("t").set_index("t")["r_m"].reindex(R1.index)
        B1 = BETA.reindex(R1.index, method="ffill").reindex(columns=R1.columns)
        E1 = R1.sub(B1.mul(rm1, axis=0))
        E1 = E1.reindex(columns=cols)

        # peer means on the 1h grid, leave-one-out
        peer1 = pd.DataFrame(np.nan, index=E1.index, columns=cols)
        for s in sec.unique():
            mem = [c for c in cols if sec[c] == s]
            if len(mem) < 2:
                continue
            blk = E1[mem]
            tot, cnt = blk.sum(axis=1, min_count=1), blk.notna().sum(axis=1)
            for c in mem:
                peer1[c] = (tot - blk[c].fillna(0.0)) / \
                           (cnt - blk[c].notna()).replace(0, np.nan)

        f3_h = _roll_corr(E1, peer1, W90D_H, MIN90D_H)
        f3 = _at(f3_h, t_obs)

        # F10: current coupling against this coin's own longer-run coupling.
        f10 = _at(_roll_corr(E1, peer1, W30D_H, MIN30D_H) - f3_h, t_obs)

        # F8: coupling to own sector minus coupling to everything else.
        rest = pd.DataFrame(np.nan, index=E1.index, columns=cols)
        for c in cols:
            others = [x for x in cols if sec[x] != sec[c]]
            rest[c] = E1[others].mean(axis=1)
        f8 = _at(f3_h - _roll_corr(E1, rest, W90D_H, MIN90D_H), t_obs)

        # F6: mean pairwise coupling INSIDE each sector (per-sector tier).
        f6 = pd.DataFrame(np.nan, index=idx, columns=cols)
        for s in sec.unique():
            mem = [c for c in cols if sec[c] == s]
            if len(mem) < 2:
                continue
            pairs = [_roll_corr(E1[[a]], E1[[b]].rename(columns={b: a}),
                                W90D_H, MIN90D_H)[a]
                     for i, a in enumerate(mem) for b in mem[i + 1:]]
            coh = _at(pd.concat(pairs, axis=1).mean(axis=1).to_frame("v"), t_obs)["v"]
            for c in mem:
                f6[c] = coh

        # F7: spread of residuals inside the sector (per-sector tier).
        f7 = pd.DataFrame(np.nan, index=idx, columns=cols)
        for s in sec.unique():
            mem = [c for c in cols if sec[c] == s]
            if len(mem) < 2:
                continue
            d = e8[mem].std(axis=1)
            for c in mem:
                f7[c] = d

        # F5: raw coupling to BTC, 8h returns, 90d -- NOT residuals, so the
        # sum-to-zero artifact (FEATURE_EXPLORATION §9.2b) does not arise.
        R8 = P / P.shift(1) - 1.0
        f5 = R8.rolling(270, min_periods=162).corr(R8["BTCUSDT"])

        # F4/F9: bar-level propagation legs. Declared as context columns --
        # identical across coins, so zero standalone ranking power.
        f4 = pd.DataFrame(np.repeat(e8["BTCUSDT"].shift(1).to_numpy()[:, None],
                                    len(cols), axis=1), index=idx, columns=cols)
        f9 = pd.DataFrame(np.repeat(e8["ETHUSDT"].shift(1).to_numpy()[:, None],
                                    len(cols), axis=1), index=idx, columns=cols)

        # F11: how much of this bar's dispersion comes from this coin.
        dev = e8.sub(e8.mean(axis=1), axis=0).abs()
        f11 = dev.div(dev.sum(axis=1, min_count=1), axis=0)

        out = {
            "sector_rel_reversal_8h": f1, "sector_rel_agreement": f2,
            "coin_sector_corr": f3, "btc_resid_lag": f4,
            "corr_btc_8h_90d": f5, "sector_cohesion": f6,
            "sector_dispersion": f7, "sector_belonging": f8,
            "eth_resid_lag": f9, "sector_decoupling": f10,
            "dispersion_contribution": f11,
        }
        res = pd.concat({k: v.stack() for k, v in out.items()}, axis=1)
        res.index = res.index.set_names(["t_obs", "symbol"])
        return res.replace([np.inf, -np.inf], np.nan).sort_index()


def _roll_corr(a: pd.DataFrame, b: pd.DataFrame, win: int,
               minp: int) -> pd.DataFrame:
    return a.rolling(win, min_periods=minp).corr(b)


def _at(hourly: pd.DataFrame, t_obs: np.ndarray) -> pd.DataFrame:
    """Sample an hourly frame at the decision instants, as-of."""
    return hourly.reindex(hourly.index.union(t_obs)).ffill().reindex(t_obs)
