"""Alternative book constructions on the SAME signal. Exploratory only.

READ THIS BEFORE READING ANY NUMBER BELOW
------------------------------------------
Both holdouts are spent. Every period this script reports has been seen. So
nothing here is a validated result and nothing here may be presented as one:
whatever wins is a HYPOTHESIS whose only honest test is forward data that does
not yet exist.

The reason to run it anyway is that the diagnosis points somewhere specific.
The signal did not decay -- IC was +0.049 on train, +0.048 in 2025, +0.039 in
2026, and hit rate actually ROSE to 55.0% in the year the book lost money. What
inverted was the magnitude of wins against losses:

    win/loss size ratio     2025  1.063     2026  0.805

Ranking worked; the translation of ranking into positions did not. That is a
property of the BOOK, and the book is the one part of this project that was
never varied. So the question is whether a different construction on the
identical signal would have survived, and it is answerable.

THE VARIANTS, declared before running, each motivated by the diagnosis rather
than chosen from a search:

    A current        rank/sigma weights over all 20 coins. The baseline.
    B top5 equal     equal weight on the best 5 and worst 5, nothing between.
                     If the signal is strongest at the extremes, spreading it
                     over the middle 10 adds noise and no edge.
    C no inverse-vol rank weights, NOT divided by sigma. Inverse-vol sizing is
                     what created the beta leak, and it systematically buys the
                     calm names -- which is the wrong side of a payoff that
                     depends on dispersion.
    D long-only top5 no shorts. Tests how much of the P&L was the short book.
    E disp-scaled    current weights, gross scaled by trailing cross-sectional
                     dispersion. Dispersion fell 30% from train to 2026 and the
                     strategy eats dispersion; this trades smaller when there
                     is less to win.

Five variants over three periods is fifteen numbers. That is a small enough
grid that the best of it is not automatically noise -- but it is not zero
either, and the caveat at the top stands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits, strategy
from mft.paths import DATA_DIR
from scripts.run_strategy import load_panel, attach_funding, CACHE
from scripts.stage7_importance import LGB_PARAMS, SEEDS

CAP_2026 = int(pd.Timestamp("2026-07-31", tz="UTC").timestamp() * 1000)
BY = strategy.BARS_PER_YEAR
K = 5


def weights(P: pd.DataFrame, kind: str) -> pd.Series:
    g = P.groupby(level="t_obs")
    n = g["score"].transform("size")
    up = g["score"].rank(ascending=False)
    dn = g["score"].rank(ascending=True)
    z = g["score"].rank(pct=True)
    z = z - z.groupby(level="t_obs").transform("mean")

    if kind == "A_current":
        w = z / P["sigma_eps"].replace(0.0, np.nan)
    elif kind == "C_no_invvol":
        w = z.copy()
    elif kind == "B_top5":
        w = pd.Series(0.0, index=P.index)
        w[up <= K] = 1.0
        w[dn <= K] = -1.0
    elif kind == "D_longonly":
        w = pd.Series(0.0, index=P.index)
        w[up <= K] = 1.0
    elif kind == "E_dispscaled":
        w = z / P["sigma_eps"].replace(0.0, np.nan)
    else:
        raise ValueError(kind)

    w = w - w.groupby(level="t_obs").transform("mean") if kind != "D_longonly" else w
    gross = w.abs().groupby(level="t_obs").transform("sum")
    w = w / gross.replace(0.0, np.nan)

    if kind == "E_dispscaled":
        # Trailing cross-sectional dispersion of the TARGET, past bars only.
        d = P.groupby(level="t_obs")["target"].std()
        s = (d.rolling(90, min_periods=30).mean().shift(1))
        s = (s / s.median()).clip(0.3, 1.5).reindex(
            P.index.get_level_values("t_obs")).to_numpy()
        w = w * np.nan_to_num(s, nan=1.0)
    return w


def run_book(P: pd.DataFrame, kind: str) -> dict:
    w = weights(P, kind).fillna(0.0)
    W = w.unstack("symbol").fillna(0.0)
    tob = W.index
    R = P["fwd_ret"].unstack("symbol").reindex(tob).reindex(columns=W.columns)
    Fn = P["funding"].unstack("symbol").reindex(tob).reindex(columns=W.columns)
    rm = P.groupby(level="t_obs")["fwd_rm"].first().reindex(tob)
    net = (W * R.fillna(0.0)).sum(axis=1) - (W * Fn.fillna(0.0)).sum(axis=1)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    turn.iloc[0] = W.abs().sum(axis=1).iloc[0]

    # same rolling-90 hedge overlay the adopted spec uses
    rr, mm = net.to_numpy(), rm.to_numpy()
    hb = np.zeros(len(rr))
    for i in range(len(rr)):
        if i >= 90:
            a, b = mm[i - 90:i], rr[i - 90:i]
            v = a.var()
            hb[i] = (np.cov(a, b)[0, 1] / v) if v > 0 else 0.0
    net = net - pd.Series(hb, index=tob) * rm

    mu, sd = float(net.mean()), float(net.std(ddof=1))
    eq = (1 + net).cumprod()
    d = pd.DataFrame({"r": net, "rm": rm}).dropna()
    cc = float(np.corrcoef(d["r"], d["rm"])[0, 1])
    return {"sharpe": mu / sd * np.sqrt(BY) if sd else np.nan,
            "ann": float((1 + mu) ** BY - 1),
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "hit": float((net > 0).mean()),
            "turn": float(turn.mean()),
            "corr_rm": cc,
            "wl": float(abs(net[net > 0].mean() / net[net < 0].mean()))}


def holdout(lo, hi):
    import lightgbm as lgb
    Xtr, Ttr, _, _ = load_panel(lo)
    Xho, Tho, _, _ = load_panel(hi, lower_ms=lo)
    Rtr = Xtr.groupby(level="t_obs").rank(pct=True)
    Rho = Xho.groupby(level="t_obs").rank(pct=True)
    Ytr = Ttr["target"].groupby(level="t_obs").rank(pct=True)
    acc = np.zeros(len(Xho))
    for sd in SEEDS:
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
            Rtr.to_numpy("float32"), Ytr.to_numpy())
        p = m.predict(Rho.to_numpy("float32"))
        acc += (p - p.mean()) / (p.std() or 1.0)
    return attach_funding(Tho.assign(score=acc / len(SEEDS)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/salvage.json")
    args = ap.parse_args()

    periods = {
        "train": pd.read_parquet(CACHE),
        "2025": holdout(splits.TRUE_HOLDOUT1_MS, splits.TRUE_HOLDOUT2_MS),
        "2026": holdout(splits.TRUE_HOLDOUT2_MS, CAP_2026),
    }
    for k, P in periods.items():
        if "target" not in P.columns:
            T = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet") \
                .set_index(["t_obs", "symbol"])["target"]
            periods[k] = P.join(T, how="left")

    kinds = ["A_current", "B_top5", "C_no_invvol", "D_longonly", "E_dispscaled"]
    print("ALTERNATIVE BOOKS ON THE IDENTICAL SIGNAL")
    print("  EXPLORATORY. Both holdouts are spent; nothing here is validated.\n")
    print(f"  {'variant':<14} " + "".join(f"{p:>22}" for p in periods))
    print(f"  {'':<14} " + "".join(f"{'Sharpe   ann   w/l':>22}" for p in periods))
    print("  " + "-" * 80)
    res = {}
    for kind in kinds:
        row = {}
        line = f"  {kind:<14} "
        for pname, P in periods.items():
            r = run_book(P, kind)
            row[pname] = r
            line += f"{r['sharpe']:>+8.2f}{r['ann']:>8.1%}{r['wl']:>6.2f}"
        res[kind] = row
        print(line)

    print("\n  detail, 2026 (the year that decided it):")
    print(f"  {'variant':<14} {'Sharpe':>8} {'ann':>8} {'maxDD':>8} {'hit':>7} "
          f"{'turn':>7} {'corr(r_m)':>10}")
    for kind in kinds:
        r = res[kind]["2026"]
        print(f"  {kind:<14} {r['sharpe']:>+8.2f} {r['ann']:>8.1%} "
              f"{r['maxdd']:>+8.1%} {r['hit']:>7.1%} {r['turn']:>7.1%} "
              f"{r['corr_rm']:>+10.4f}")

    print("\n" + "=" * 82)
    pos = {k: sum(1 for p in periods if res[k][p]["sharpe"] > 0) for k in kinds}
    print("  positive in all three periods: " +
          (", ".join(k for k, v in pos.items() if v == 3) or "NONE"))
    print("  A variant that is positive in three periods it was CHOSEN on is")
    print("  not evidence. It is a candidate for forward testing, nothing more.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
