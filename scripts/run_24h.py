"""The 24h experiment. Train through 2025, test on 2026.

WHAT IS AND IS NOT NEW HERE
----------------------------
NEW: the horizon. 24h instead of 8h, pre-registered in TARGET_DESIGN.md §2 as
"a documented alternative ... if 8h underperforms."

UNCHANGED: the 72 frozen features, the model, the seed ensemble, the book, the
rolling-90 hedge. Features are the EXISTING 8h panel restricted to 00:00 rows.
Daily instants are a strict subset of 8h instants, so no feature is recomputed
and no window changes meaning -- which matters, because every feature window is
a BAR COUNT and rebuilding on a daily grid would silently turn "30 days" into
"90 days". That bug class has already appeared three times in this codebase
(anatomy.py's bar_duration, the W30D windows, and min_obs in the weights).

HOW MUCH THIS TEST IS WORTH, STATED UP FRONT
---------------------------------------------
Less than it looks. 2026 has been seen: the 8h book was run on it and lost. A
24h return is three compounded 8h returns over the same prices and coins, so
the two are heavily overlapping and my knowledge of one contaminates the other.

This is therefore NOT a clean out-of-sample test. It is a re-examination of
already-spent data under a different horizon, and its honest status is
"suggestive". The only clean test remaining is forward data.

Train is everything before 2026-01-01: 1,822 daily instants. That is close to
the ~1,820 TARGET_DESIGN.md §2 predicted for 24h, and about a third of the 8h
sample -- the cost the design doc named when it chose 8h.
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
from scripts.stage7_importance import LGB_PARAMS, SEEDS, FAMILIES, per_bar_ic
from scripts.backtest_train import funding_per_window

CAP = int(pd.Timestamp("2026-07-31", tz="UTC").timestamp() * 1000)
BY_24 = 365          # one bar per day


def load():
    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])
    X = pd.concat([pd.read_parquet(DATA_DIR / f"features/{f}.parquet")
                   .set_index(["t_obs", "symbol"]) for f in FAMILIES.values()
                   if (DATA_DIR / f"features/{f}.parquet").exists()], axis=1)[cols]
    T = pd.read_parquet(DATA_DIR / "grid/target_24h_lag60.parquet") \
        .set_index(["t_obs", "symbol"])
    i = X.index.intersection(T.index)          # restricts X to daily instants
    return X.loc[i].sort_index(), T.loc[i].sort_index(), cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/run_24h.json")
    args = ap.parse_args()
    import lightgbm as lgb

    X, T, cols = load()
    t = X.index.get_level_values("t_obs").to_numpy()
    tr = t < splits.TRUE_HOLDOUT2_MS
    te = (t >= splits.TRUE_HOLDOUT2_MS) & (t < CAP)
    print(f"24h EXPERIMENT -- {len(cols)} frozen features, daily bars")
    print(f"  train {int(tr.sum()):,} rows / {len(np.unique(t[tr])):,} bars "
          f"({splits._fmt(int(t[tr].min()))} .. {splits._fmt(int(t[tr].max()))})")
    print(f"  test  {int(te.sum()):,} rows / {len(np.unique(t[te])):,} bars "
          f"({splits._fmt(int(t[te].min()))} .. {splits._fmt(int(t[te].max()))})\n")

    Rk = X.groupby(level="t_obs").rank(pct=True)
    Yr = T["target"].groupby(level="t_obs").rank(pct=True)
    acc = np.zeros(int(te.sum()))
    for sd in SEEDS:
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
            Rk[tr].to_numpy("float32"), Yr[tr].to_numpy())
        p = m.predict(Rk[te].to_numpy("float32"))
        acc += (p - p.mean()) / (p.std() or 1.0)
    score = pd.Series(acc / len(SEEDS), index=X.index[te])

    P = T[te].assign(score=score)
    tob = np.sort(P.index.get_level_values("t_obs").unique())
    f = funding_per_window(tob, sorted(P.index.get_level_values("symbol").unique()))
    f = f.stack()
    f.index = f.index.set_names(["t_obs", "symbol"])
    P = P.join(f.rename("funding"), how="left")
    P["funding"] = P["funding"].fillna(0.0)

    ic = per_bar_ic(P["score"].to_numpy(), P["target"].to_numpy(),
                    P.index.get_level_values("t_obs").to_numpy())
    print(f"  2026 IC (24h horizon): {ic:+.4f}")

    old_by = strategy.BARS_PER_YEAR
    strategy.BARS_PER_YEAR = BY_24               # one bar per day, not three
    try:
        res = strategy.run(P)
        r = res.ret
        idx = pd.to_datetime(r.index, unit="ms")
        per = pd.DataFrame({"r": r.to_numpy(), "t": res.turnover.to_numpy()},
                           index=idx)
        print("\nMONTHLY (24h book, 2026)")
        print("=" * 74)
        print(f"  {'month':<9} {'bars':>5} {'return':>9} {'Sharpe':>8} "
              f"{'vol':>8} {'maxDD':>8} {'hit':>7} {'turn':>7}")
        rows = []
        for mo, gdf in per.groupby(per.index.to_period("M")):
            mm = strategy.metrics(gdf["r"], turnover=float(gdf["t"].mean()))
            cum = float((1 + gdf["r"]).prod() - 1)
            rows.append({"month": str(mo), **{k: mm[k] for k in
                        ("bars", "sharpe", "ann_vol", "max_drawdown",
                         "hit_rate", "turnover")}, "return": cum})
            print(f"  {str(mo):<9} {mm['bars']:>5} {cum:>+9.2%} "
                  f"{mm['sharpe']:>+8.2f} {mm['ann_vol']:>8.1%} "
                  f"{mm['max_drawdown']:>+8.2%} {mm['hit_rate']:>7.1%} "
                  f"{mm['turnover']:>7.1%}")
        pos = sum(1 for x in rows if x["return"] > 0)
        m = strategy.metrics(r, turnover=float(res.turnover.mean()))
        print(f"\n  {pos}/{len(rows)} months positive")
        print("\nFULL 2026")
        print("=" * 74)
        print(f"  ann return {m['ann_return']:+.2%}   vol {m['ann_vol']:.2%}   "
              f"Sharpe {m['sharpe']:+.2f}")
        print(f"  hit {m['hit_rate']:.1%}   maxDD {m['max_drawdown']:+.2%}   "
              f"{m['bars']:,} bars   turnover {m['turnover']:.1%}")
        rm = P.groupby(level="t_obs")["fwd_rm"].first().reindex(r.index)
        d = pd.DataFrame({"r": r, "rm": rm, "a": rm.abs()}).dropna()
        for lbl, c in [("r_m  ", "rm"), ("|r_m|", "a")]:
            cc = float(np.corrcoef(d["r"], d[c])[0, 1])
            tt = cc * np.sqrt(len(d) - 2) / np.sqrt(max(1e-12, 1 - cc * cc))
            print(f"  corr(book, {lbl}) {cc:+.4f}  t {tt:+.2f}")
    finally:
        strategy.BARS_PER_YEAR = old_by

    print("\n  8h book on the same period, for reference: Sharpe -0.35")
    print("  2026 was already seen. This is suggestive, not out-of-sample.")
    out = DATA_DIR / args.out
    out.write_text(json.dumps({"ic_2026": float(ic), "metrics": m,
                               "monthly": rows}, indent=2, default=float),
                   encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
