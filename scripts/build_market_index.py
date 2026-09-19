"""Step 2 -- the market factor r_m.

Implements Step 2 of docs/TARGET_DESIGN.md. Turns Step 1's weights into the
volume-weighted index return series that beta is measured against (D1 of
docs/BETA_NEUTRAL_DESIGN.md).

    r_i,t  =  px_fill(t) / px_fill(t-1) - 1        fill-to-fill, one 8h step
    r_m,t  =  sum_i  w'_i,(t-1) * r_i,t            weights from the PREVIOUS step

Two things this gets right that are easy to get wrong:

LAGGED WEIGHTS. r_m at step t uses weights stamped at t-1. Those were built
from volume observed before t_obs(t-1), which is before the return window even
opens at t_fill(t-1). Using step-t weights would let the index know which coins
traded heavily during the very period it is measuring -- a classic lookahead
that inflates an index's apparent quality.

GAPS ARE NOT RETURNS. The 2022 archive gap leaves two discontinuities where
consecutive ROWS are 3.7 and 2.3 days apart. Differencing rows blindly would
book a multi-day move as a single 8h return -- a huge fake outlier landing in
every beta window that spans it. A return is only formed when the previous
observation is exactly one step earlier, per symbol.

w' is w renormalised over the coins that actually have a return at t, so a
missing coin redistributes its weight rather than silently shrinking the index.

    python scripts/build_market_index.py
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
FAILURES: list[str] = []

# Pre-registered in docs/TARGET_DESIGN.md Gate 2, before any number was seen.
CORR_BTC_MAX = 0.99   # at or above: the index IS BTC, D1 gains nothing
CORR_BTC_MIN = 0.75   # below: warrants investigation
MIN_WEIGHT_COVERAGE = 0.90


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def fmt(ms) -> str:
    return splits._fmt(int(ms))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="grid/decision_grid_8h.parquet")
    ap.add_argument("--weights", default="grid/market_weights_30d.parquet")
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--out", default="grid/market_index_8h.parquet")
    args = ap.parse_args()

    step_ms = args.h * MS_HOUR
    panel = pd.read_parquet(DATA_DIR / args.grid)
    wts = pd.read_parquet(DATA_DIR / args.weights)
    out_path = DATA_DIR / args.out

    print(f"Step 2 -- market index  h={args.h}h")
    print(f"  grid    {len(panel):,} rows, {panel.t_obs.nunique():,} cross-sections")
    print(f"  weights {len(wts):,} rows, {wts.t_obs.nunique():,} cross-sections\n")

    # ---- per-coin fill-to-fill returns, gap-aware ------------------------
    p = panel.sort_values(["symbol", "t_obs"]).copy()
    g = p.groupby("symbol", sort=False)
    p["t_prev"] = g["t_obs"].shift(1)
    p["px_prev"] = g["px_fill"].shift(1)
    # Only a genuinely consecutive observation makes a one-step return.
    p["consecutive"] = (p["t_obs"] - p["t_prev"]) == step_ms
    p["ret"] = np.where(p["consecutive"], p["px_fill"] / p["px_prev"] - 1.0, np.nan)

    n_skipped_gap = int(((p["t_prev"].notna()) & (~p["consecutive"])).sum())
    rets = p.dropna(subset=["ret"])[["symbol", "t_obs", "ret"]]

    # ---- attach the PREVIOUS step's weight -------------------------------
    # Shift the weight forward one step in time so it is applied to the return
    # that follows it, then join on the return's own stamp.
    w = wts[["symbol", "t_obs", "weight"]].sort_values(["symbol", "t_obs"]).copy()
    w["t_apply"] = w["t_obs"] + step_ms          # weight known at t_obs governs step t_obs+h
    w = w.rename(columns={"t_obs": "t_weight"})

    m = rets.merge(w, left_on=["symbol", "t_obs"], right_on=["symbol", "t_apply"],
                   how="inner", validate="one_to_one")

    # ---- renormalise and aggregate ---------------------------------------
    cov = m.groupby("t_obs")["weight"].sum().rename("coverage")
    m = m.merge(cov, on="t_obs")
    m["w_norm"] = m["weight"] / m["coverage"]
    m["contrib"] = m["w_norm"] * m["ret"]

    idx = m.groupby("t_obs").agg(
        r_m=("contrib", "sum"),
        n_coins=("symbol", "size"),
        coverage=("coverage", "first"),
        t_weight=("t_weight", "first"),
    ).reset_index()

    thin_cov = idx[idx["coverage"] < MIN_WEIGHT_COVERAGE]
    idx = idx[idx["coverage"] >= MIN_WEIGHT_COVERAGE].reset_index(drop=True)

    # ---- reference series for the gate -----------------------------------
    btc = rets[rets.symbol == "BTCUSDT"][["t_obs", "ret"]].rename(columns={"ret": "r_btc"})
    eth = rets[rets.symbol == "ETHUSDT"][["t_obs", "ret"]].rename(columns={"ret": "r_eth"})
    ew = rets.groupby("t_obs")["ret"].mean().rename("r_ew").reset_index()
    cmp = idx.merge(btc, on="t_obs").merge(eth, on="t_obs").merge(ew, on="t_obs")

    c_btc = float(cmp["r_m"].corr(cmp["r_btc"]))
    c_eth = float(cmp["r_m"].corr(cmp["r_eth"]))
    c_ew = float(cmp["r_m"].corr(cmp["r_ew"]))
    c_ew_btc = float(cmp["r_ew"].corr(cmp["r_btc"]))

    # ---- GATE 2 -----------------------------------------------------------
    print("GATE 2\n" + "=" * 74)

    # The lag, asserted rather than assumed: every weight must predate the
    # return window it is applied to.
    gate(bool((m["t_weight"] < m["t_obs"]).all()),
         "every weight stamp strictly precedes the return it weights",
         f"lag {int((m.t_obs - m.t_weight).min()) / MS_HOUR:.0f}h uniform"
         if len(m) else "")
    gate(bool(((m["t_obs"] - m["t_weight"]) == step_ms).all()),
         f"weight lag is exactly one step ({args.h}h)")

    gate(n_skipped_gap > 0,
         "gap-spanning returns detected and excluded",
         f"{n_skipped_gap} (symbol, step) pairs skipped across the 2022 holes")
    max_span = int((p.loc[p["consecutive"], "t_obs"] - p.loc[p["consecutive"], "t_prev"]).max())
    gate(max_span == step_ms, "no return spans more than one step",
         f"max {max_span / MS_HOUR:.0f}h")

    gate(bool(np.isfinite(idx["r_m"]).all()), "index return finite everywhere")
    gate(bool((idx["coverage"] >= MIN_WEIGHT_COVERAGE).all()),
         f"weight coverage >= {MIN_WEIGHT_COVERAGE:.0%} on every retained step",
         f"min {idx.coverage.min():.3f}, {len(thin_cov)} steps dropped")
    try:
        splits.assert_train_only(idx.assign(bar_close_time=idx["t_obs"]))
        gate(True, "splits.assert_train_only(t_obs)")
    except splits.HoldoutLeak as e:
        gate(False, "splits.assert_train_only(t_obs)", str(e)[:70])

    # The pre-registered correlation band.
    gate(c_btc < CORR_BTC_MAX,
         f"corr(r_m, r_BTC) < {CORR_BTC_MAX} -- index is not merely BTC",
         f"corr = {c_btc:.4f}")
    gate(c_btc > CORR_BTC_MIN, f"corr(r_m, r_BTC) > {CORR_BTC_MIN}",
         f"corr = {c_btc:.4f}")

    print("\nCOMPARISONS (reported)\n" + "=" * 74)
    print(f"  corr(r_m, r_BTC)          {c_btc:.4f}   <- Gate 2 band "
          f"[{CORR_BTC_MIN}, {CORR_BTC_MAX})")
    print(f"  corr(r_m, r_ETH)          {c_eth:.4f}")
    print(f"  corr(r_m, r_equalweight)  {c_ew:.4f}")
    print(f"  corr(r_equalweight, BTC)  {c_ew_btc:.4f}   <- what an EW index would give")

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"GATE 2 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        print("Nothing written. Per TARGET_DESIGN.md: revisit the weighting (D1).")
        return 1

    # ---- census -----------------------------------------------------------
    print("\nCENSUS\n" + "=" * 74)
    ann = np.sqrt(365 * 24 / args.h)
    print(f"  steps {len(idx):,}   {fmt(idx.t_obs.min())} -> {fmt(idx.t_obs.max())}")
    print(f"  coins per step: mean {idx.n_coins.mean():.1f}  min {int(idx.n_coins.min())}"
          f"  max {int(idx.n_coins.max())}")
    print(f"  r_m: mean {idx.r_m.mean():+.5f}  std {idx.r_m.std():.5f}"
          f"  -> annualised vol {idx.r_m.std()*ann:.1%}")
    print(f"  r_m range: {idx.r_m.min():+.2%} .. {idx.r_m.max():+.2%}")

    cmp2 = cmp.assign(y=pd.to_datetime(cmp.t_obs, unit="ms", utc=True).dt.year)
    print(f"\n  {'year':<6} {'steps':>6} {'corr BTC':>9} {'r_m vol':>9} {'BTC vol':>9} "
          f"{'r_m cum':>9} {'BTC cum':>9}")
    for y, gg in cmp2.groupby("y"):
        print(f"  {y:<6} {len(gg):>6,} {gg.r_m.corr(gg.r_btc):>9.4f} "
              f"{gg.r_m.std()*ann:>8.1%} {gg.r_btc.std()*ann:>8.1%} "
              f"{(1+gg.r_m).prod()-1:>8.1%} {(1+gg.r_btc).prod()-1:>8.1%}")

    # ---- write ------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx[["t_obs", "t_weight", "r_m", "n_coins", "coverage"]].to_parquet(out_path, index=False)

    manifest = {
        "step": 2,
        "source_grid": args.grid,
        "source_weights": args.weights,
        "h_hours": args.h,
        "return_basis": "px_fill to px_fill, one step, gap-aware",
        "weight_lag_steps": 1,
        "min_weight_coverage": MIN_WEIGHT_COVERAGE,
        "steps": int(len(idx)),
        "gap_spanning_returns_excluded": int(n_skipped_gap),
        "low_coverage_steps_dropped": int(len(thin_cov)),
        "corr_r_m_btc": round(c_btc, 6),
        "corr_r_m_eth": round(c_eth, 6),
        "corr_r_m_equalweight": round(c_ew, 6),
        "corr_equalweight_btc": round(c_ew_btc, 6),
        "gate2_band": [CORR_BTC_MIN, CORR_BTC_MAX],
        "annualised_vol": round(float(idx.r_m.std() * ann), 6),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2),
                                                      encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"GATE 2 PASSED. Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
