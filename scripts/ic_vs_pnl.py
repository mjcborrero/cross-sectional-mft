"""Why does a POSITIVE cross-sectional IC produce a NEGATIVE book?

scripts/ridge.py found ridge holding IC +0.036 (train) -> +0.031 (2026) while
its 2026 book returned -15%. Predictive power that survives out-of-sample and
still loses money is not an overfitting story, so the defect is downstream of
the prediction. This isolates where.

THE SUSPECT
-----------
The target is  (fwd_ret - beta*fwd_rm) / sigma_eps  -- a VOL-NORMALISED
residual. Ranking it correctly is not the same as making money, because the
book trades DOLLARS, not vol units. A model can rank the normalised quantity
well while its correct calls sit in low-sigma coins (small dollar payoff) and
its wrong calls sit in high-sigma coins (large dollar loss). The rank IC is
blind to that; the P&L is not.

So the IC is recomputed in three spaces:

    target      (fwd_ret - beta*fwd_rm) / sigma_eps   <- what is optimised
    residual     fwd_ret - beta*fwd_rm                <- dollars, beta-neutral
    raw          fwd_ret                              <- dollars, unhedged

If IC(target) > 0 while IC(residual) <= 0, the normalisation is where the edge
evaporates, and the target -- not the model -- is the thing to change.

THE SECOND SUSPECT
------------------
Mean IC is an average over all coins. The book is top-k/bottom-k, so it only
ever holds the TAILS. A positive average IC is compatible with losing tails.
The decile spread in dollar terms tests that directly.
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
from mft import splits, strategy
from mft.paths import DATA_DIR
from scripts.ridge import ALPHAS, CAP, fit_predict, ranks
from scripts.run_strategy import cache_path, load_panel
from scripts.stage7_importance import per_bar_ic

K = 5           # the book's top-k / bottom-k


def spaces(T: pd.DataFrame) -> dict[str, np.ndarray]:
    resid = (T["fwd_ret"] - T["beta"] * T["fwd_rm"]).to_numpy()
    return {"target (vol-normalised)": T["target"].to_numpy(),
            "residual (dollars)": resid,
            "raw fwd_ret (dollars)": T["fwd_ret"].to_numpy()}


def tail_spread(score: np.ndarray, T: pd.DataFrame) -> dict:
    """What the BOOK actually earns: mean dollar residual of the top-k long
    leg minus the bottom-k short leg, per bar."""
    d = pd.DataFrame({
        "s": score, "b": T.index.get_level_values("t_obs"),
        "res": (T["fwd_ret"] - T["beta"] * T["fwd_rm"]).to_numpy(),
        "raw": T["fwd_ret"].to_numpy()})
    out = {}
    for col in ("res", "raw"):
        longs, shorts = [], []
        for _, g in d.groupby("b"):
            if len(g) < 2 * K:
                continue
            o = g.sort_values("s")
            shorts.append(o[col].iloc[:K].mean())
            longs.append(o[col].iloc[-K:].mean())
        L, S = np.array(longs), np.array(shorts)
        out[col] = {"long_bps": float(L.mean() * 1e4),
                    "short_bps": float(S.mean() * 1e4),
                    "spread_bps": float((L - S).mean() * 1e4),
                    "bars": len(L)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1000.0)
    ap.add_argument("--out", default="features/ic_vs_pnl.json")
    a = ap.parse_args()

    lo = splits.TRUE_HOLDOUT2_MS
    Xtr, Ttr, cols, _ = load_panel(lo, lag=a.lag)
    Xte, Tte, _, _ = load_panel(CAP, lower_ms=lo, lag=a.lag)
    Rtr, _ = ranks(Xtr)
    Rte, _ = ranks(Xte)
    Ytr = Ttr["target"].groupby(level="t_obs").rank(pct=True).to_numpy()

    models = {}
    p = fit_predict(Rtr, Ytr, Rte, a.alpha)
    models[f"ridge a={a.alpha:g}"] = pd.Series(
        (p - p.mean()) / (p.std() or 1.0), index=Xte.index)

    # LightGBM, refit the same way so the comparison is like-for-like.
    import lightgbm as lgb
    from scripts.stage7_importance import LGB_PARAMS, SEEDS
    acc = np.zeros(len(Xte))
    for sd in SEEDS:
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
            Rtr.to_numpy("float32"), Ytr)
        q = m.predict(Rte.to_numpy("float32"))
        acc += (q - q.mean()) / (q.std() or 1.0)
    models["lightgbm (8 seeds)"] = pd.Series(acc / len(SEEDS), index=Xte.index)

    bar = Tte.index.get_level_values("t_obs").to_numpy()
    print("2026 -- IC MEASURED IN THREE SPACES")
    print("  the book trades dollars; the target is vol-normalised.\n")
    print(f"  {'model':<20}" + "".join(f"{k:>26}" for k in spaces(Tte)))
    res = {}
    for name, sc in models.items():
        row, line = {}, f"  {name:<20}"
        for label, y in spaces(Tte).items():
            ic = per_bar_ic(sc.to_numpy(), y, bar)
            row[label] = ic
            line += f"{ic:>+26.4f}"
        res[name] = {"ic": row}
        print(line)

    print("\n\nWHAT THE BOOK ACTUALLY HOLDS -- top/bottom "
          f"{K} legs, mean per bar (bps)")
    print(f"  {'model':<20}{'space':<12}{'long':>9}{'short':>9}"
          f"{'spread':>10}{'bars':>7}")
    for name, sc in models.items():
        ts = tail_spread(sc.to_numpy(), Tte)
        res[name]["tails"] = ts
        for col, lbl in (("res", "residual"), ("raw", "fwd_ret")):
            t = ts[col]
            print(f"  {name:<20}{lbl:<12}{t['long_bps']:>+9.2f}"
                  f"{t['short_bps']:>+9.2f}{t['spread_bps']:>+10.2f}"
                  f"{t['bars']:>7}")

    print("\n  A POSITIVE spread means the ranking makes money in that space")
    print("  before the book's own construction (neutrality, hedge, turnover).")

    out = DATA_DIR / a.out
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
