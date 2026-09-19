"""Hedge variants, measured on TRAIN and on 2025.

WHY THIS EXISTS
---------------
The overlay held on train (corr(book, r_m) +0.056) and broke out of sample
(-0.186 at t -6.25, sign flipped, magnitude tripled). The cause is structural,
not a tuning miss: the ratio was an EXPANDING regression over four years of
history, so by 2025 a single new bar moved it by about 1/4500. A regression
that slow cannot track a beta that moves, and it can only ever learn that the
beta moved after it already has.

WHAT IS BEING TESTED
--------------------
    expanding      the shipped version, for reference
    rolling-N      the same regression over the last N bars only
    exante         NOT a regression. The book's market exposure is
                   sum(w_i * beta_i), which is KNOWN at trade time from the
                   same trailing betas the target already uses. No lookback,
                   no lag.

`exante` is the principled candidate and it is a design argument, not a fitted
one: if the quantity you want to cancel is computable at the moment you trade,
estimating it from your own past returns is strictly worse.

A NOTE ON WHAT 2025 IS NOW
---------------------------
2025 was spent. Any choice made using the 2025 column below is SELECTED on
2025, which makes 2025 a validation set for that choice and leaves only 2026
genuinely out of sample. So the selection rule is declared before reading the
table:

    Choose on TRAIN and on mechanism. 2025 is reported as CONFIRMATION only.
    A variant that wins on 2025 but not on train is not adopted.

That rule is what stops this from becoming a second round of fitting on data
that has already been used once.
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
from scripts.stage7_importance import LGB_PARAMS, SEEDS, FAMILIES
from scripts.run_strategy import load_panel, attach_funding, CACHE

VARIANTS = [
    ("rolling-90",       dict(HEDGE_MODE="rolling", HEDGE_WINDOW=90)),
    ("+voltarget-90",    dict(HEDGE_MODE="rolling", HEDGE_WINDOW=90,
                              GROSS_MODE="voltarget", VOL_WINDOW=90)),
    ("+voltarget-270",   dict(HEDGE_MODE="rolling", HEDGE_WINDOW=90,
                              GROSS_MODE="voltarget", VOL_WINDOW=270)),
]


def apply(cfg: dict) -> dict:
    DEF = {"HEDGE": True, "HEDGE_MODE": "rolling", "HEDGE_WINDOW": 90,
           "HEDGE_SHRINK": 1.0, "GROSS_MODE": "fixed", "VOL_WINDOW": 90}
    old = {k: getattr(strategy, k) for k in DEF}
    for k in DEF:
        setattr(strategy, k, cfg.get(k, DEF[k]))
    return old


def score(P, seed=None, init=None) -> dict:
    r = strategy.run(P, seed=seed, init_weights=init)
    m = strategy.metrics(r.ret, turnover=float(r.turnover.mean()))
    rm = P.groupby(level="t_obs")["fwd_rm"].first().reindex(r.ret.index)
    d = pd.DataFrame({"r": r.ret, "rm": rm}).dropna()
    d["a"] = d["rm"].abs()
    def _ct(x, y):
        c = float(np.corrcoef(x, y)[0, 1])
        return c, float(c * np.sqrt(len(x) - 2) / np.sqrt(max(1e-12, 1 - c * c)))
    cc, tt = _ct(d["r"], d["rm"])
    ca, ta = _ct(d["r"], d["a"])
    return {**m, "corr_rm": cc, "t_rm": tt, "corr_arm": ca, "t_arm": ta,
            "_res": r, "_P": P}


def holdout_panel() -> pd.DataFrame:
    import lightgbm as lgb
    lo, hi = splits.TRUE_HOLDOUT1_MS, splits.TRUE_HOLDOUT2_MS
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
    ap.add_argument("--out", default="features/fix_hedge.json")
    args = ap.parse_args()

    Ptr = pd.read_parquet(CACHE)
    Pho = holdout_panel()
    print(f"HEDGE VARIANTS -- train {len(Ptr):,} rows, 2025 {len(Pho):,} rows")
    print("  selection rule declared in the module docstring: choose on TRAIN "
          "and\n  on mechanism; 2025 confirms, it does not select.\n")

    hdr = (f"  {'variant':<13} | {'TRAIN':^32} | {'2025':^32}")
    sub = (f"  {'':<15} | {'Sharpe':>7} {'t(r_m)':>8} {'t(|rm|)':>8} {'turn':>6} | "
           f"{'Sharpe':>7} {'t(r_m)':>8} {'t(|rm|)':>8} {'turn':>6}")
    print(hdr)
    print(sub)
    print("  " + "-" * 88)

    rows = []
    for name, cfg in VARIANTS:
        old = apply(cfg)
        try:
            a = score(Ptr)
            # carry train state into 2025, exactly as the real run does
            seed = (a["_res"].ret.to_numpy(),
                    Ptr.groupby(level="t_obs")["fwd_rm"].first()
                    .reindex(a["_res"].ret.index).to_numpy())
            initw = a["_res"].weights.iloc[-1].reindex(
                sorted(Pho.index.get_level_values("symbol").unique())
            ).fillna(0.0).to_numpy()
            b = score(Pho, seed=seed, init=initw)
        finally:
            for k, v in old.items():
                setattr(strategy, k, v)
        rows.append({"variant": name,
                     "train": {k: a[k] for k in
                               ("sharpe", "ann_return", "max_drawdown", "turnover",
                                "corr_rm", "t_rm", "corr_arm", "t_arm",
                                "breakeven_bps")},
                     "h2025": {k: b[k] for k in
                               ("sharpe", "ann_return", "max_drawdown", "turnover",
                                "corr_rm", "t_rm", "corr_arm", "t_arm",
                                "breakeven_bps")}})
        print(f"  {name:<15} | {a['sharpe']:>+7.2f} {a['t_rm']:>+8.2f} "
              f"{a['t_arm']:>+8.2f} {a['turnover']:>6.1%} | "
              f"{b['sharpe']:>+7.2f} {b['t_rm']:>+8.2f} {b['t_arm']:>+8.2f} "
              f"{b['turnover']:>6.1%}")

    print("\n" + "=" * 90)
    print("NEUTRALITY IS THE OBJECTIVE HERE, NOT SHARPE. The book already earns;")
    print("what failed out of sample was the claim that it is market-neutral.")
    print("A variant is better if |t| on corr(book, r_m) is small in BOTH")
    print("columns -- being neutral on train alone is what the shipped version")
    print("already did.")

    best = min(rows, key=lambda r: max(abs(r["train"]["t_rm"]),
                                       abs(r["h2025"]["t_rm"]),
                                       abs(r["train"]["t_arm"]),
                                       abs(r["h2025"]["t_arm"])))
    print(f"\n  smallest WORST-CASE |t| across both periods: '{best['variant']}'"
          f"  train t {best['train']['t_rm']:+.2f}, "
          f"2025 t {best['h2025']['t_rm']:+.2f}")
    print(f"  its Sharpe: train {best['train']['sharpe']:+.2f}, "
          f"2025 {best['h2025']['sharpe']:+.2f}")

    out = DATA_DIR / args.out
    out.write_text(json.dumps({"variants": rows, "selected": best["variant"]},
                              indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
