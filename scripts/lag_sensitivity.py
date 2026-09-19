"""Step 7L -- how much does execution delay cost?

Implements Step 7L / Gate 3L of docs/TARGET_DESIGN.md.

There is no model and no feature set yet, so a genuine "does the edge survive"
test is not available. Claiming one would be theatre. Two things ARE
measurable now, and together they bound the answer:

1. LABEL DRIFT (model-free). corr(target at lag 5, target at lag L). If the
   label barely changes, no model's IC could move much either -- this bounds
   the damage without needing a model.

   Baseline to judge against: two windows of length T overlapping by O have
   correlation O/T under serially independent returns. lag 60 vs lag 5 overlap
   by 7h55m of 8h = 0.9896. Materially BELOW that means delay destroys
   something real; at or above means the label is merely being shifted.

2. PROBE SIGNAL. Cross-sectional short-horizon reversal -- computable from
   prices alone, no feature engineering, and the most robust known effect at
   this horizon. Its IC against the target at each lag shows whether a real
   short-horizon effect decays with delay. It is a PROBE, not the strategy:
   its own sign and size say nothing about what a fitted model would achieve.

The probe signal uses px_obs (observation prices), so it is fully known at
t_obs -- unlike fill-to-fill returns, whose previous window closes at
t_obs+lag, after the decision.

    python scripts/lag_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits
from mft.paths import DATA_DIR

MS_HOUR = 3_600_000
H = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="grid/forward_8h_lagsweep.parquet")
    ap.add_argument("--betas", default="grid/beta_idiovol_8h.parquet")
    ap.add_argument("--grid", default="grid/decision_grid_8h.parquet")
    ap.add_argument("--base-lag", type=int, default=5)
    args = ap.parse_args()

    sweep = pd.read_parquet(DATA_DIR / args.sweep)
    bi = pd.read_parquet(DATA_DIR / args.betas)[
        ["symbol", "t_obs", "beta", "sigma_eps"]]
    dec = pd.read_parquet(DATA_DIR / args.grid)
    lags = sorted(sweep["lag_min"].unique())

    print(f"Step 7L -- lag sensitivity   lags {lags} min\n")

    # Provisional target assembly (Step 8 does the gated version). beta and
    # sigma_eps are stamped at t_obs and do not depend on lag, so only the
    # forward legs move.
    m = sweep.merge(bi, on=["symbol", "t_obs"], how="inner")
    m["target"] = (m["fwd_ret"] - m["beta"] * m["fwd_rm"]) / m["sigma_eps"]

    wide = m.pivot_table(index=["t_obs", "symbol"], columns="lag_min",
                         values="target")
    wide = wide.dropna()
    base = args.base_lag

    # ---- 1. label drift ---------------------------------------------------
    print("1. LABEL DRIFT -- how much does the target itself change?\n" + "=" * 74)
    print(f"  {'lag':>6} {'corr vs lag5':>13} {'IID baseline':>13} {'vs baseline':>12}"
          f" {'std':>8}")
    drift = {}
    for L in lags:
        overlap = (H * 60 - (L - base)) / (H * 60)
        c = float(wide[base].corr(wide[L]))
        drift[L] = {"corr": c, "baseline": overlap}
        flag = "" if L == base else f"{c - overlap:+.4f}"
        print(f"  {L:>4}m  {c:>13.4f} {overlap:>13.4f} {flag:>12} "
              f"{wide[L].std():>8.4f}")

    # ---- 2. probe signal --------------------------------------------------
    # Previous observation-to-observation residual return, known at t_obs.
    p = dec.sort_values(["symbol", "t_obs"]).copy()
    g = p.groupby("symbol", sort=False)
    p["t_prev"] = g["t_obs"].shift(1)
    p["px_obs_prev"] = g["px_obs"].shift(1)
    p = p[(p["t_obs"] - p["t_prev"]) == H * MS_HOUR]
    p["r_prev"] = p["px_obs"] / p["px_obs_prev"] - 1.0

    # Cross-sectional market leg for the same window, then residualise with the
    # same beta the target uses, so the probe is not a disguised beta bet.
    mk = p.groupby("t_obs")["r_prev"].mean().rename("rm_prev")
    p = p.merge(mk, on="t_obs").merge(bi, on=["symbol", "t_obs"], how="inner")
    p["signal"] = -(p["r_prev"] - p["beta"] * p["rm_prev"]) / p["sigma_eps"]

    print("\n2. PROBE SIGNAL -- cross-sectional reversal, IC by lag\n" + "=" * 74)
    print(f"  {'lag':>6} {'mean IC':>9} {'std err':>9} {'t-stat':>8} {'bars':>7}"
          f" {'vs lag5':>9}")
    probe = {}
    ic5 = None
    for L in lags:
        tgt = m[m["lag_min"] == L][["symbol", "t_obs", "target"]]
        j = p[["symbol", "t_obs", "signal"]].merge(tgt, on=["symbol", "t_obs"])
        per_bar = (j.groupby("t_obs")
                     .apply(lambda d: d["signal"].corr(d["target"], method="spearman")
                            if len(d) >= 10 else np.nan, include_groups=False)
                     .dropna())
        mu, se = float(per_bar.mean()), float(per_bar.std() / np.sqrt(len(per_bar)))
        if L == base:
            ic5 = mu
        probe[L] = {"ic": mu, "se": se, "n_bars": int(len(per_bar))}
        rel = "" if L == base else f"{mu/ic5:.3f}x" if ic5 else ""
        print(f"  {L:>4}m  {mu:>9.4f} {se:>9.4f} {mu/se:>8.2f} {len(per_bar):>7,}"
              f" {rel:>9}")

    # ---- read-out ---------------------------------------------------------
    c60 = drift[60]["corr"]
    b60 = drift[60]["baseline"]
    ic60, ic_base = probe[60]["ic"], probe[base]["ic"]
    retained = ic60 / ic_base if ic_base else float("nan")

    print("\nREAD-OUT\n" + "=" * 74)
    print(f"  Label drift at lag 60: corr {c60:.4f} vs IID baseline {b60:.4f}"
          f"  ({c60 - b60:+.4f})")
    if c60 >= b60 - 0.01:
        print("    -> the label is essentially just shifted; delay is not")
        print("       destroying information beyond simple window overlap.")
    else:
        print("    -> the label degrades FASTER than overlap alone; short-horizon")
        print("       structure is being lost to delay.")

    print(f"\n  Probe IC retained at 60min: {retained:.1%} of the lag-5 value"
          f"  ({ic_base:+.4f} -> {ic60:+.4f})")
    sig60 = abs(probe[60]["ic"] / probe[60]["se"])
    if retained > 0.8 and sig60 > 2:
        verdict = "lag 60min is viable; a real short-horizon effect survives it"
    elif retained > 0.5:
        verdict = "lag 60min costs materially but does not erase the probe effect"
    else:
        verdict = ("lag 60min erases most of the probe effect -- an edge at this "
                   "horizon may not be capturable at pipeline speed")
    print(f"  -> {verdict}")

    print("\n  Caveat, on the record: the probe is short-horizon reversal, not the")
    print("  eventual model. It bounds decay for one known effect. A model built")
    print("  on slower mechanisms would decay less; one on microstructure, more.")

    out = DATA_DIR / "grid" / "lag_sensitivity.json"
    out.write_text(json.dumps({
        "step": "7L", "lags_min": [int(x) for x in lags], "base_lag": base,
        "label_drift": {str(k): {kk: round(float(vv), 6) for kk, vv in v.items()}
                        for k, v in drift.items()},
        "probe_reversal_ic": {str(k): {kk: (round(float(vv), 6)
                                            if kk != "n_bars" else vv)
                                       for kk, vv in v.items()}
                              for k, v in probe.items()},
        "probe_ic_retained_at_60": round(float(retained), 5),
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
