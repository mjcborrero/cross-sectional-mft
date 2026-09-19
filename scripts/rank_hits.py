"""Does the ridge RANKING work, measured as per-coin hits -- not as IC.

WHAT IS ASKED
-------------
Rank the 20 coins with ridge at each decision bar. Hold for H days, H in
{2,3,4,5}. The top 10 are longs, the bottom 10 are shorts. For every coin in
every ranking record a tuple:

    (hit, return)     hit = 1 if the position made money, else 0

A long hits if its return is > 0; a short hits if its return is < 0. The full
tuple table is saved so a second model can be built on top of the rank. The
point is to see the ranking as an object, not through the lens of a single
correlation number.

THE LABEL IS THE SAME ONE, AT A LONGER HORIZON
-----------------------------------------------
    target_H = (fwd_ret_H - beta * fwd_rm_H) / (sigma_eps * sqrt(k))

fwd_ret_H is k consecutive 8h forward returns COMPOUNDED, k = 3H. The 8h
return at bar t+1 starts exactly where bar t's ends (t_fill + 8h), so the
chain is gap-free by construction. This is the method build_target_24h.py
used, cross-checked against an independent daily build at 8.88e-16. beta and
sigma_eps are the 8h panel's values at t_obs; sigma scales by sqrt(k), a common
multiplier that leaves within-bar ranks untouched.

The rolling window runs on a DENSE 8h grid, not on row count. The 8h grid has
17 interior instants missing (thin cross-sections), and a bar-count window
would silently stretch across them. This bug class has appeared three times in
this codebase.

RETURN IS REPORTED IN TWO SPACES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
--------------------------------------------------------------------------
    resid   fwd_ret - beta*fwd_rm    what a beta-neutral book earns; the label
    raw     fwd_ret                  "did the coin go up"

A ranking can be good at one and useless at the other. In a bull market every
long "goes up" and raw hit rates flatter the ranking; the BASE RATE (hit rate
of a random assignment) is printed beside each so the lift is visible.

WITH 20 COINS, TOP 10 + BOTTOM 10 IS EVERY COIN
------------------------------------------------
So the side-level hit rates are not "does the model pick well" so much as "is
the median split informative". The rank-position table (1..20) is where the
shape of the ranking shows: if rank 20 is not better than rank 11, the ordering
inside the long half carries nothing, and a picker built on it has nothing to
pick from.

HONESTY OF THE RANKING
----------------------
Ridge scores are OUT-OF-FOLD from purged walk-forward folds on TRAIN only
(before 2025). The purge is widened to ceil((24H + 1)/8) bars for each H --
16 bars at 5 days, not the 8h default of 2. Statistics use NON-OVERLAPPING
bars (every k-th) so consecutive rankings do not share their forward window;
the saved table keeps every bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import folds as foldmod
from mft import splits
from mft.paths import DATA_DIR
from scripts.ridge import fit_predict, ranks
from scripts.run_strategy import load_panel

STEP_MS = 8 * 3_600_000
HORIZONS_D = [2, 3, 4, 5]          # days; --bars overrides in 8h bars
TOP = 10


def compound_forward(T: pd.DataFrame, k: int) -> pd.DataFrame:
    """k consecutive 8h forward returns compounded, on a DENSE 8h grid."""
    out = {}
    for col in ("fwd_ret", "fwd_rm"):
        W = T[col].unstack("symbol")
        full = np.arange(W.index.min(), W.index.max() + STEP_MS, STEP_MS)
        W = W.reindex(full)
        # window of k bars STARTING at t: roll forward, then shift back
        C = np.log1p(W).rolling(k, min_periods=k).sum().shift(-(k - 1))
        out[col] = np.expm1(C).stack(future_stack=True)
    R = pd.DataFrame(out).dropna()
    R.index = R.index.set_names(["t_obs", "symbol"])
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1000.0)
    ap.add_argument("--out", default="features/rank_hits")
    ap.add_argument("--bars", default=None,
                    help="comma list of horizons in 8h BARS (e.g. '1' for the "
                         "8h book, '1,3' for 8h and 24h). Overrides the "
                         "2-5 day default.")
    ap.add_argument("--model", default="ridge", choices=["ridge", "lgbm"],
                    help="ridge (closed form) or lgbm (the frozen 8-seed "
                         "ensemble). Same folds, same label, same table.")
    a = ap.parse_args()
    if a.model == "lgbm":
        import lightgbm as lgb
        from scripts.stage7_importance import LGB_PARAMS, SEEDS

        def fit_predict_model(Rtr, ytr, Rte, _alpha):
            acc = np.zeros(len(Rte))
            for sd in SEEDS:
                m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
                    Rtr.to_numpy("float32"), ytr)
                q = m.predict(Rte.to_numpy("float32"))
                acc += (q - q.mean()) / (q.std() or 1.0)
            return acc / len(SEEDS)
    else:
        fit_predict_model = fit_predict

    X, T8, cols, _ = load_panel(splits.TRUE_HOLDOUT1_MS, lag=a.lag)
    bars = X.index.get_level_values("t_obs").to_numpy()
    ubars = np.sort(np.unique(bars))
    R, fill = ranks(X)
    print(f"RANK HITS -- {a.model}{' a=%g' % a.alpha if a.model == 'ridge' else ' (8 seeds)'}, {len(cols)} features, "
          f"TRAIN only ({len(ubars):,} bars, neutral-filled {fill:.2%})")
    print(f"  top {TOP} = long, bottom {TOP} = short, out-of-fold rankings\n")

    summary, tables = {}, []
    ks = ([int(x) for x in a.bars.split(",")] if a.bars
          else [3 * H for H in HORIZONS_D])
    for k in ks:
        H = k / 3                                   # days, for labels
        htag = f"{8*k}h" if k < 3 else f"{H:g}d"
        F = compound_forward(T8, k).join(T8[["beta", "sigma_eps"]], how="inner")
        F["resid"] = F["fwd_ret"] - F["beta"] * F["fwd_rm"]
        F["target"] = F["resid"] / (F["sigma_eps"] * np.sqrt(k))
        idx = X.index.intersection(F.index)
        Xh, Fh, Rh = X.loc[idx], F.loc[idx], R.loc[idx]
        bh = Xh.index.get_level_values("t_obs").to_numpy()
        Yh = Fh["target"].groupby(level="t_obs").rank(pct=True).to_numpy()

        # Folds are laid out on the FULL 8h bar set, not on the bars this
        # horizon has left. Compounding drops the last k-1 bars (their forward
        # window runs past the data), and the fold module's fixed
        # 1803 + 5x551 layout would refuse the shortfall. Defining folds on
        # `ubars` with the purge widened for this H, then intersecting, keeps
        # the fold boundaries IDENTICAL across horizons -- so the four tables
        # below are comparable -- and simply leaves absent bars out of the
        # last test block. make_folds reads PURGE_BARS at call time.
        foldmod.PURGE_BARS = int(np.ceil((8 * k + 1) / 8))
        sc = pd.Series(np.nan, index=Xh.index)
        for f in foldmod.make_folds(ubars):
            tr, te = np.isin(bh, f.train), np.isin(bh, f.test)
            p = fit_predict_model(Rh[tr], Yh[tr], Rh[te], a.alpha)
            sc[te] = (p - p.mean()) / (p.std() or 1.0)

        D = Fh.assign(score=sc).dropna(subset=["score"])
        D["rank"] = D.groupby(level="t_obs")["score"].rank(method="first").astype(int)
        n = D.groupby(level="t_obs")["rank"].transform("max")
        D = D[n == 20]                                  # full cross-sections only
        D["side"] = np.where(D["rank"] > 20 - TOP, "long",
                             np.where(D["rank"] <= TOP, "short", "none"))
        D = D[D["side"] != "none"]
        sgn = np.where(D["side"] == "long", 1.0, -1.0)
        D["ret_resid"] = D["resid"] * sgn          # signed: profit if > 0
        D["ret_raw"] = D["fwd_ret"] * sgn
        D["hit_resid"] = (D["ret_resid"] > 0).astype(int)
        D["hit_raw"] = (D["ret_raw"] > 0).astype(int)
        D["H_days"] = H

        # ---- statistics on NON-OVERLAPPING bars only ---------------------
        tb = np.sort(D.index.get_level_values("t_obs").unique())[::k]
        S = D[D.index.get_level_values("t_obs").isin(tb)]
        base_resid = float((S["resid"] > 0).mean())      # random long
        base_raw = float((S["fwd_ret"] > 0).mean())

        print("=" * 78)
        print(f"H = {htag}  (k={k} bars, purge {foldmod.PURGE_BARS}, "
              f"{len(tb):,} non-overlapping rankings, {len(S):,} positions)")
        print("=" * 78)
        print(f"  {'side':<7}{'n':>6} | {'hit resid':>10}{'base':>7}{'lift':>7}"
              f"{'mean bps':>10} | {'hit raw':>9}{'base':>7}{'lift':>7}{'mean bps':>10}")
        row = {}
        for side, g in S.groupby("side"):
            br = base_resid if side == "long" else 1 - base_resid
            bw = base_raw if side == "long" else 1 - base_raw
            hr, hw = g["hit_resid"].mean(), g["hit_raw"].mean()
            print(f"  {side:<7}{len(g):>6} | {hr:>10.1%}{br:>7.1%}{hr-br:>+7.1%}"
                  f"{g['ret_resid'].mean()*1e4:>+10.1f} | {hw:>9.1%}{bw:>7.1%}"
                  f"{hw-bw:>+7.1%}{g['ret_raw'].mean()*1e4:>+10.1f}")
            row[side] = {"n": len(g), "hit_resid": hr, "base_resid": br,
                         "hit_raw": hw, "base_raw": bw,
                         "mean_resid_bps": float(g["ret_resid"].mean() * 1e4),
                         "mean_raw_bps": float(g["ret_raw"].mean() * 1e4)}

        print(f"\n  BY RANK POSITION  (1 = most negative score = strongest short)")
        print(f"  {'rank':>5}{'side':>7}{'hit resid':>11}{'mean bps':>10}"
              f"  {'hit raw':>9}{'mean bps':>10}")
        pr = S.groupby("rank").agg(side=("side", "first"),
                                   hit_resid=("hit_resid", "mean"),
                                   mean_resid=("ret_resid", "mean"),
                                   hit_raw=("hit_raw", "mean"),
                                   mean_raw=("ret_raw", "mean"))
        for r, g in pr.iterrows():
            mark = "  <" if r in (1, 20) else ""
            print(f"  {r:>5}{g['side']:>7}{g['hit_resid']:>11.1%}"
                  f"{g['mean_resid']*1e4:>+10.1f}  {g['hit_raw']:>9.1%}"
                  f"{g['mean_raw']*1e4:>+10.1f}{mark}")
        ext = S[S["rank"].isin([1, 2, 19, 20])]["hit_resid"].mean()
        mid = S[S["rank"].isin([9, 10, 11, 12])]["hit_resid"].mean()
        print(f"  extremes (1,2,19,20) hit {ext:.1%}   "
              f"middle (9-12) hit {mid:.1%}   gap {ext-mid:+.1%}")

        print(f"\n  BY COIN  (hit rate on resid when long / when short)")
        pc = S.groupby(["symbol", "side"])["hit_resid"].agg(["mean", "size"]).unstack("side")
        pc.columns = [f"{a}_{b}" for a, b in pc.columns]
        pc = pc.sort_values("mean_long", ascending=False)
        print(f"  {'coin':<10}{'long hit':>9}{'n':>5}{'short hit':>11}{'n':>5}")
        for sym, g in pc.iterrows():
            print(f"  {sym.replace('USDT',''):<10}{g['mean_long']:>9.1%}"
                  f"{int(g['size_long']):>5}{g['mean_short']:>11.1%}"
                  f"{int(g['size_short']):>5}")
        print()

        row["rank_position"] = {int(r): {"hit_resid": float(g["hit_resid"]),
                                         "mean_resid_bps": float(g["mean_resid"] * 1e4)}
                                for r, g in pr.iterrows()}
        row["extremes_vs_middle"] = {"extremes": float(ext), "middle": float(mid)}
        summary[htag] = row
        tables.append(D.reset_index())

    full = pd.concat(tables, ignore_index=True)
    cols_out = ["H_days", "t_obs", "symbol", "rank", "side", "score",
                "hit_resid", "ret_resid", "hit_raw", "ret_raw",
                "fwd_ret", "fwd_rm", "beta", "sigma_eps"]
    pq = DATA_DIR / f"{a.out}.parquet"
    full[cols_out].to_parquet(pq, index=False)
    (DATA_DIR / f"{a.out}.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("=" * 78)
    print(f"  full tuple table (every bar, all horizons): {len(full):,} rows")
    print(f"  -> {pq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
