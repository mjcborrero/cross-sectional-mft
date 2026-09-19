"""Step 1 -- point-in-time volume weights for the market index.

Implements Step 1 of docs/TARGET_DESIGN.md. Produces the weights that Step 2
turns into the market factor r_m (D1 of docs/BETA_NEUTRAL_DESIGN.md: the
volume-weighted index of the tradeable universe).

    w_i,t  =  adv_i,t / sum_j adv_j,t
    adv_i,t = mean 8h quote volume over (t_obs - window, t_obs]

Why that window is point-in-time despite including the current row: the grid's
`qv_window` at t_obs covers [t_obs - 8h, t_obs), so it is already trailing data.
Nothing in adv_i,t is observed at or after t_obs.

A mean (not a sum) over the trailing window, so the 17 cross-sections lost to
the 2022 archive gap degrade a coin's weight gracefully instead of halving it
for a month.

Market cap is not available from public Binance data, so volume is the
weighting variable. It also matches the design intent -- the index should
represent the market of the coins actually traded, and volume is what makes a
coin tradeable.

Step 2 applies the one-step LAG (r_m at step t uses w from step t-1). This
script does not lag; it stamps each weight with the instant it became knowable.

    python scripts/build_market_weights.py
    python scripts/build_market_weights.py --window-days 60
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

FAILURES: list[str] = []


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def fmt(ms) -> str:
    return splits._fmt(int(ms))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="grid/decision_grid_8h.parquet")
    ap.add_argument("--window-days", type=int, default=30,
                    help="trailing volume window, days (default 30)")
    ap.add_argument("--min-obs", type=int, default=30,
                    help="min 8h windows required before a coin is weighted "
                         "(default 30 = 10 days)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grid_path = DATA_DIR / args.grid
    panel = pd.read_parquet(grid_path)
    out_path = DATA_DIR / (args.out or f"grid/market_weights_{args.window_days}d.parquet")

    print(f"Step 1 -- market weights  window={args.window_days}d  min_obs={args.min_obs}")
    print(f"Source: {grid_path}")
    print(f"  {len(panel):,} rows  {panel.symbol.nunique()} symbols  "
          f"{panel.t_obs.nunique():,} cross-sections\n")

    # ---- trailing average dollar volume, per symbol ----------------------
    panel = panel.sort_values(["symbol", "t_obs"]).copy()
    panel["dt"] = pd.to_datetime(panel["t_obs"], unit="ms", utc=True)

    parts = []
    for sym, g in panel.groupby("symbol", sort=False):
        s = g.set_index("dt")["qv_window"]
        # Time-based window, so gaps shorten the sample rather than shifting it.
        adv = s.rolling(f"{args.window_days}D", min_periods=args.min_obs).mean()
        parts.append(pd.DataFrame({
            "symbol": sym,
            "t_obs": g["t_obs"].to_numpy(),
            "adv": adv.to_numpy(),
            "n_obs": s.rolling(f"{args.window_days}D", min_periods=1).count().to_numpy(),
        }))
    w = pd.concat(parts, ignore_index=True)

    weighted = w.dropna(subset=["adv"]).copy()
    weighted = weighted[weighted["adv"] > 0]

    # ---- normalise within each cross-section -----------------------------
    tot = weighted.groupby("t_obs")["adv"].transform("sum")
    weighted["weight"] = weighted["adv"] / tot
    weighted = weighted.sort_values(["t_obs", "symbol"]).reset_index(drop=True)

    # Cross-sections where nobody has enough history yet are simply absent.
    covered = weighted.t_obs.nunique()
    warmup = panel.t_obs.nunique() - covered

    # ---- GATE 1 -----------------------------------------------------------
    print("GATE 1\n" + "=" * 74)
    sums = weighted.groupby("t_obs")["weight"].sum()
    gate(bool(np.allclose(sums.to_numpy(), 1.0, atol=1e-12)),
         "weights sum to 1 in every cross-section",
         f"max deviation {abs(sums - 1).max():.2e}")

    gate(bool((weighted["weight"] > 0).all()), "all weights strictly positive")
    gate(bool((weighted["weight"] <= 1).all()), "no weight exceeds 1")
    gate(not weighted[["adv", "weight"]].isna().any().any(), "no NaN in adv or weight")

    # No coin may carry weight before it had data.
    first_seen = panel.groupby("symbol")["t_obs"].min()
    first_wt = weighted.groupby("symbol")["t_obs"].min()
    early = [s for s in first_wt.index if first_wt[s] < first_seen[s]]
    gate(not early, "no coin weighted before its first observation",
         f"{early}" if early else f"{len(first_wt)} symbols")

    # Every weighted coin met the minimum-history bar.
    gate(bool((weighted["n_obs"] >= args.min_obs).all()),
         f"every weighted coin has >= {args.min_obs} trailing observations",
         f"min {int(weighted.n_obs.min())}")

    # Point-in-time: adv at t_obs is built from qv_window values that each
    # close at or before t_obs, so no input is observed after the stamp.
    gate(bool((weighted["t_obs"].isin(set(panel["t_obs"]))).all()),
         "every weight instant exists in the decision grid")
    try:
        splits.assert_train_only(weighted.assign(bar_close_time=weighted["t_obs"]))
        gate(True, "splits.assert_train_only(t_obs)")
    except splits.HoldoutLeak as e:
        gate(False, "splits.assert_train_only(t_obs)", str(e)[:70])

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"GATE 1 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        print("Nothing written.")
        return 1

    # ---- reported diagnostics --------------------------------------------
    print("\nCENSUS\n" + "=" * 74)
    print(f"  weighted rows {len(weighted):,}   cross-sections {covered:,}"
          f"   ({warmup} dropped to {args.window_days}d warm-up)")
    print(f"  {fmt(weighted.t_obs.min())} -> {fmt(weighted.t_obs.max())}")

    mean_w = weighted.groupby("symbol")["weight"].agg(["mean", "min", "max"])
    mean_w = mean_w.sort_values("mean", ascending=False)
    print(f"\n  {'symbol':<10} {'mean w':>8} {'min':>8} {'max':>8}")
    for s, r in mean_w.iterrows():
        print(f"  {s:<10} {r['mean']:>8.3f} {r['min']:>8.3f} {r['max']:>8.3f}")

    piv = weighted.pivot(index="t_obs", columns="symbol", values="weight").fillna(0.0)
    hhi = (piv ** 2).sum(axis=1)
    eff_n = 1.0 / hhi
    print(f"\n  concentration:  effective N  mean {eff_n.mean():.1f}  "
          f"min {eff_n.min():.1f}  max {eff_n.max():.1f}   (of {panel.symbol.nunique()} coins)")

    yr = piv.copy()
    yr.index = pd.to_datetime(yr.index, unit="ms", utc=True).year
    print(f"\n  BTC / ETH weight and effective N by year:")
    print(f"    {'year':<6} {'BTC':>7} {'ETH':>7} {'effN':>7}")
    for y, g in yr.groupby(level=0):
        e = 1.0 / (g ** 2).sum(axis=1).mean()
        print(f"    {y:<6} {g['BTCUSDT'].mean():>7.3f} {g['ETHUSDT'].mean():>7.3f} {e:>7.1f}")

    # ---- write ------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weighted[["symbol", "t_obs", "adv", "n_obs", "weight"]].to_parquet(out_path, index=False)

    manifest = {
        "step": 1,
        "source_grid": str(args.grid),
        "window_days": args.window_days,
        "min_obs": args.min_obs,
        "weighting_variable": "quote_volume (USDT), trailing mean of 8h windows",
        "lag_applied": False,
        "lag_note": "Step 2 uses w from step t-1 when forming r_m at step t",
        "rows": int(len(weighted)),
        "cross_sections": int(covered),
        "warmup_cross_sections_dropped": int(warmup),
        "t_obs_first_ms": int(weighted.t_obs.min()),
        "t_obs_last_ms": int(weighted.t_obs.max()),
        "mean_weight": {s: round(float(v), 5) for s, v in mean_w["mean"].items()},
        "effective_n_mean": round(float(eff_n.mean()), 2),
    }
    mpath = out_path.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"GATE 1 PASSED. Wrote {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")
    print(f"Manifest: {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
