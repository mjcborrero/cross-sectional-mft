"""Step 4 -- at what return frequency should beta be estimated?

Implements Step 4 of docs/TARGET_DESIGN.md.

The hazard: non-synchronous trading biases high-frequency covariances toward
zero (the Epps effect), so betas estimated on very short returns come out too
LOW -- and differentially so, worst for the least-traded coins. That is not a
harmless level effect: it is a cross-sectional distortion, and the target
divides by a quantity derived from beta.

    beta_i(f) = cov(r_i(f), r_m(f)) / var(r_m(f))

estimated at f in {1h, 2h, 4h, 8h, 24h}, each coin over its own fixed window
so the coin is its own control and only the sampling frequency changes.

The index is rebuilt at every frequency from the same volume weights, asof-
matched so the weight in force always predates the return window it weights.

GATE 4: adopt the finest frequency at which beta has stabilised -- mean |beta|
changes less than 5% versus the next-coarser frequency.

The sharpest test is not the level but the SLOPE against liquidity: if Epps is
present, illiquid coins' betas should rise more than liquid ones' as the
frequency coarsens. That correlation is reported, because a uniform shift and
a liquidity-graded shift have different implications.

    python scripts/beta_frequency.py
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
from mft.data.store import ParquetStore
from mft.paths import DATA_DIR, NORMALIZED_DIR

MS_MIN = 60_000
MS_HOUR = 3_600_000
FAILURES: list[str] = []

FREQS = [("1h", 1), ("2h", 2), ("4h", 4), ("8h", 8), ("24h", 24)]
STABILITY_TOL = 0.05          # pre-registered: "less than 5%"
TOL_MS = 5 * MS_MIN           # same staleness rule as Step 0


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def asof_px(ct: np.ndarray, px: np.ndarray, instants: np.ndarray) -> np.ndarray:
    """Last close strictly before each instant; NaN if staler than TOL_MS."""
    i = np.searchsorted(ct, instants, side="left") - 1
    out = np.full(len(instants), np.nan)
    ok = i >= 0
    if ok.any():
        v = i[ok]
        age = instants[ok] - ct[v]
        fresh = age <= TOL_MS
        out[np.where(ok)[0][fresh]] = px[v][fresh]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="grid/market_weights_30d.parquet")
    ap.add_argument("--start", default="2021-01-01",
                    help="common analysis window start (default 2021-01-01)")
    ap.add_argument("--universe", default="config/universe.txt")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    universe = [l.strip().upper() for l in
                (repo / args.universe).read_text(encoding="utf-8-sig").splitlines()
                if l.strip() and not l.startswith("#")]

    wall = splits.HOLDOUT1_START_MS
    lo = int(pd.Timestamp(args.start, tz="UTC").value // 10**6)
    wts = pd.read_parquet(DATA_DIR / args.weights)

    print(f"Step 4 -- beta estimation frequency (Epps check)")
    print(f"  window {args.start} -> {splits._fmt(wall)}   {len(universe)} coins\n")

    # ---- prices on every frequency grid ----------------------------------
    store = ParquetStore(NORMALIZED_DIR)
    grids = {name: np.arange(-(-lo // (k * MS_HOUR)) * k * MS_HOUR, wall, k * MS_HOUR,
                             dtype=np.int64) for name, k in FREQS}
    prices = {name: {} for name, _ in FREQS}
    advs = {}

    for sym in universe:
        kl = store.load("perp_klines", sym, columns=["close_time", "close"])
        kl = kl[kl["close_time"] < wall].sort_values("close_time")
        ct = kl["close_time"].to_numpy()
        px = kl["close"].to_numpy()
        for name, _ in FREQS:
            prices[name][sym] = asof_px(ct, px, grids[name])
        w = wts[wts.symbol == sym]
        advs[sym] = float(w["adv"].mean()) if len(w) else np.nan

    # ---- per-frequency returns, index, and betas -------------------------
    rows = []
    per_freq = {}
    for name, k in FREQS:
        step = k * MS_HOUR
        gr = grids[name]
        P = pd.DataFrame(prices[name], index=gr)
        R = P.pct_change()
        # A return is only valid if BOTH endpoints exist; pct_change already
        # yields NaN otherwise, and the grid is regular so there are no
        # hidden multi-step jumps.
        R = R.replace([np.inf, -np.inf], np.nan)

        # Weights in force at the START of each return window, asof-matched so
        # the weight always predates what it weights.
        wq = (wts[["t_obs", "symbol", "weight"]]
              .pivot(index="t_obs", columns="symbol", values="weight"))
        W = wq.reindex(wq.index.union(gr)).ffill().reindex(gr)
        W = W.shift(1)                       # strictly before the window opens
        W = W.reindex(columns=R.columns)

        M = W.where(R.notna())               # only coins with a return
        M = M.div(M.sum(axis=1), axis=0)     # renormalise
        r_m = (M * R).sum(axis=1, min_count=1)
        valid = M.sum(axis=1).notna() & (M.notna().sum(axis=1) >= 10)
        r_m = r_m.where(valid)

        betas, r2s, ns = {}, {}, {}
        for sym in R.columns:
            d = pd.concat([R[sym].rename("y"), r_m.rename("x")], axis=1).dropna()
            if len(d) < 200:
                continue
            vx = d["x"].var()
            b = d["y"].cov(d["x"]) / vx
            betas[sym] = b
            r2s[sym] = d["y"].corr(d["x"]) ** 2
            ns[sym] = len(d)
        per_freq[name] = {"beta": pd.Series(betas), "r2": pd.Series(r2s),
                          "n": pd.Series(ns), "vol_m": float(r_m.std())}
        rows.append(name)

    # ---- report -----------------------------------------------------------
    B = pd.DataFrame({n: per_freq[n]["beta"] for n in rows})
    R2 = pd.DataFrame({n: per_freq[n]["r2"] for n in rows})
    B = B.dropna()

    print("BETA BY FREQUENCY (per coin)\n" + "=" * 74)
    adv_s = pd.Series({s: advs[s] for s in B.index}).sort_values()
    print(f"  {'symbol':<10} " + " ".join(f"{n:>7}" for n in rows) +
          f"  {'1h->24h':>8}  {'adv rank':>8}")
    for i, sym in enumerate(adv_s.index):
        if sym not in B.index:
            continue
        chg = B.loc[sym, "24h"] / B.loc[sym, "1h"] - 1
        print(f"  {sym:<10} " + " ".join(f"{B.loc[sym, n]:>7.3f}" for n in rows) +
              f"  {chg:>+7.1%}  {i+1:>8}")

    print("\nAGGREGATES\n" + "=" * 74)
    print(f"  {'freq':<6} {'mean beta':>10} {'std beta':>9} {'mean R2':>9} "
          f"{'obs/coin':>9} {'r_m vol':>9}")
    for n in rows:
        print(f"  {n:<6} {B[n].mean():>10.4f} {B[n].std():>9.4f} "
              f"{R2[n].mean():>9.3f} {int(per_freq[n]['n'].mean()):>9,} "
              f"{per_freq[n]['vol_m']:>9.5f}")

    # ---- GATE 4 -----------------------------------------------------------
    print("\nGATE 4  (pre-registered: <5% change vs next-coarser)\n" + "=" * 74)
    means = {n: float(B[n].mean()) for n in rows}
    stable_from = None
    print(f"  {'freq':<6} {'mean beta':>10} {'vs coarser':>12}")
    for i, n in enumerate(rows):
        if i + 1 < len(rows):
            nxt = rows[i + 1]
            chg = means[n] / means[nxt] - 1
            mark = "stable" if abs(chg) < STABILITY_TOL else "SHIFTS"
            print(f"  {n:<6} {means[n]:>10.4f} {chg:>+11.2%}  {mark}")
            if abs(chg) < STABILITY_TOL and stable_from is None:
                stable_from = n
        else:
            print(f"  {n:<6} {means[n]:>10.4f} {'--':>12}")

    gate(stable_from is not None,
         "beta stabilises at some frequency",
         f"finest stable = {stable_from}" if stable_from else "no stabilisation")

    # Epps signature: does the frequency effect grade with liquidity?
    chg = (B["24h"] / B["1h"] - 1)
    rank = pd.Series({s: advs[s] for s in B.index}).rank()
    rho = float(chg.corr(rank, method="spearman"))
    print(f"\n  Epps signature -- corr(beta change 1h->24h, liquidity rank) = {rho:+.3f}")
    print(f"    negative => illiquid coins shift MORE (classic Epps)")
    print(f"    mean |change| across coins: {chg.abs().mean():.2%}, "
          f"max {chg.abs().max():.2%} ({chg.abs().idxmax()})")

    # The weighted-mean beta is 1 by construction; verify, as an arithmetic check
    # that the index and the betas are mutually consistent.
    wmean = {}
    for n in rows:
        w_last = wts.groupby("symbol")["weight"].mean().reindex(B.index)
        w_last = w_last / w_last.sum()
        wmean[n] = float((B[n] * w_last).sum())
    print(f"\n  volume-weighted mean beta (should be ~1.0 at every frequency):")
    print("    " + "  ".join(f"{n}={wmean[n]:.3f}" for n in rows))
    gate(all(abs(v - 1.0) < 0.10 for v in wmean.values()),
         "volume-weighted mean beta ~= 1 at every frequency (index/beta consistent)",
         f"range {min(wmean.values()):.3f}-{max(wmean.values()):.3f}")

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"GATE 4 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        print("Do NOT proceed on a guessed frequency -- the bias is differential.")
        return 1

    out = DATA_DIR / "grid" / "beta_frequency.json"
    out.write_text(json.dumps({
        "step": 4, "window_start": args.start,
        "frequencies": rows,
        "mean_beta": {n: round(means[n], 5) for n in rows},
        "std_beta": {n: round(float(B[n].std()), 5) for n in rows},
        "mean_r2": {n: round(float(R2[n].mean()), 5) for n in rows},
        "weighted_mean_beta": {n: round(wmean[n], 5) for n in rows},
        "beta_by_coin": {n: {s: round(float(B.loc[s, n]), 5) for s in B.index} for n in rows},
        "epps_liquidity_rho": round(rho, 4),
        "stability_tol": STABILITY_TOL,
        "finest_stable": stable_from,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"GATE 4 PASSED. Finest stable frequency: {stable_from}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
