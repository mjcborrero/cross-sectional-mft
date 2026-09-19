"""Step 3 -- factor structure test: is one factor enough?

Implements Step 3 of docs/TARGET_DESIGN.md and the pre-registered test in
docs/BETA_NEUTRAL_DESIGN.md sec 4.

The hazard: if crypto carries a meaningful SECOND common factor (plausibly a
BTC-vs-alt or large-vs-small dimension) and the book is neutralised only to a
single market index, the book is not neutral. Factor 1 is removed and an
unhedged systematic bet on factor 2 remains -- which presents as alpha until
the factor turns.

PRE-REGISTERED DECISION RULE (stated before any number was computed):

    PC2 >= 15%                -> two-factor neutralisation; beta becomes a
                                 vector and the target gains a second term
    PC2 <  15%                -> single factor sufficient, proceed as specified
    corr(PC1, r_m) < 0.90     -> D1 is not capturing the dominant factor;
                                 return to the base design

    Expected: PC1 70-85%, PC2 5-12%, corr(PC1, r_m) > 0.95

Run on the TRAIN split only, on the same gap-aware 8h returns used in Step 2.

Two samples are reported together, never the better one:

    ALL20  every coin, listwise-complete -- short window (SUI lists 2023-05)
           but the full cross-section
    LONG   coins with near-complete history, full window -- fewer coins, but
           4+ years and every regime

If they disagree, the disagreement IS the finding. Per-year shares are also
reported, because a factor structure that moves is itself a result.

Note on PC1: it is estimated from the whole sample and is NOT point-in-time,
so it could never serve as a live factor. That is exactly why D1 uses the
observable volume-weighted index and keeps PC1 as a diagnostic only.

    python scripts/factor_structure.py
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

# Pre-registered thresholds -- docs/BETA_NEUTRAL_DESIGN.md sec 4.
PC2_TWO_FACTOR = 0.15
CORR_PC1_MIN = 0.90


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def fmt(ms) -> str:
    return splits._fmt(int(ms))


def eig_shares(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) as variance shares, plus eigenvectors."""
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return vals / vals.sum(), vecs


def analyse(R: pd.DataFrame, label: str, r_m: pd.Series) -> dict:
    """Correlation-matrix PCA on a complete return block."""
    corr = np.corrcoef(R.to_numpy(), rowvar=False)
    shares, vecs = eig_shares(corr)

    # Orient PC1 so it is the "everything moves together" factor, not its
    # mirror image -- eigenvector signs are arbitrary.
    v1 = vecs[:, 0]
    if v1.sum() < 0:
        v1, vecs = -v1, vecs * -1
    Z = (R - R.mean()) / R.std()
    pc1 = pd.Series(Z.to_numpy() @ v1, index=R.index)
    v2 = vecs[:, 1]
    pc2 = pd.Series(Z.to_numpy() @ v2, index=R.index)

    aligned = r_m.reindex(R.index)
    c1 = float(pc1.corr(aligned))
    c2 = float(pc2.corr(aligned))

    return {
        "label": label, "n_coins": R.shape[1], "n_steps": R.shape[0],
        "first": int(R.index.min()), "last": int(R.index.max()),
        "shares": shares, "pc1_loadings": pd.Series(v1, index=R.columns),
        "pc2_loadings": pd.Series(v2, index=R.columns),
        "corr_pc1_rm": c1, "corr_pc2_rm": c2,
    }


def bootstrap_shares(R: pd.DataFrame, n: int, seed: int) -> dict:
    """Resample time rows to get an interval on PC1/PC2 shares.

    8h returns are non-overlapping, so an IID row bootstrap is appropriate --
    there is no label overlap to induce serial dependence.
    """
    rng = np.random.default_rng(seed)
    X = R.to_numpy()
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(X), len(X))
        c = np.corrcoef(X[idx], rowvar=False)
        if not np.isfinite(c).all():
            continue
        s, _ = eig_shares(c)
        out.append(s[:3])
    a = np.array(out)
    return {
        "pc1": (float(np.percentile(a[:, 0], 2.5)), float(np.percentile(a[:, 0], 97.5))),
        "pc2": (float(np.percentile(a[:, 1], 2.5)), float(np.percentile(a[:, 1], 97.5))),
        "pc3": (float(np.percentile(a[:, 2], 2.5)), float(np.percentile(a[:, 2], 97.5))),
        "n": len(a),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="grid/decision_grid_8h.parquet")
    ap.add_argument("--index", default="grid/market_index_8h.parquet")
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--long-coverage", type=float, default=0.95,
                    help="coverage needed to join the LONG sample (default 0.95)")
    args = ap.parse_args()

    step_ms = args.h * MS_HOUR
    panel = pd.read_parquet(DATA_DIR / args.grid)
    idx = pd.read_parquet(DATA_DIR / args.index)
    r_m = idx.set_index("t_obs")["r_m"]

    # ---- gap-aware returns, same construction as Step 2 -------------------
    p = panel.sort_values(["symbol", "t_obs"]).copy()
    g = p.groupby("symbol", sort=False)
    p["t_prev"] = g["t_obs"].shift(1)
    p["px_prev"] = g["px_fill"].shift(1)
    p = p[(p["t_obs"] - p["t_prev"]) == step_ms]
    p["ret"] = p["px_fill"] / p["px_prev"] - 1.0
    wide = p.pivot(index="t_obs", columns="symbol", values="ret")

    print(f"Step 3 -- factor structure  ({wide.shape[0]:,} steps x "
          f"{wide.shape[1]} coins, gap-aware)")
    print(f"  {fmt(wide.index.min())} -> {fmt(wide.index.max())}\n")

    # ---- two samples, reported together ----------------------------------
    all20 = wide.dropna(axis=0, how="any")
    cov = wide.notna().mean()
    long_cols = cov[cov >= args.long_coverage].index.tolist()
    longs = wide[long_cols].dropna(axis=0, how="any")

    results = [analyse(all20, "ALL20", r_m), analyse(longs, "LONG", r_m)]

    print("SAMPLES\n" + "=" * 74)
    for r in results:
        print(f"  {r['label']:<6} {r['n_coins']:>3} coins  {r['n_steps']:>6,} steps  "
              f"{fmt(r['first'])} -> {fmt(r['last'])}")
    print(f"  LONG members ({len(long_cols)}): {', '.join(s.replace('USDT','') for s in long_cols)}")
    dropped = [s.replace("USDT", "") for s in wide.columns if s not in long_cols]
    print(f"  excluded from LONG (coverage <{args.long_coverage:.0%}): {', '.join(dropped)}")

    print("\nVARIANCE SHARES\n" + "=" * 74)
    boots = {}
    for r in results:
        s = r["shares"]
        R = all20 if r["label"] == "ALL20" else longs
        b = bootstrap_shares(R, args.boot, args.seed)
        boots[r["label"]] = b
        print(f"  {r['label']}  ({r['n_coins']} coins, {r['n_steps']:,} steps)")
        for i, name in enumerate(["PC1", "PC2", "PC3"]):
            lo, hi = b[name.lower()]
            print(f"    {name}  {s[i]:>6.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"    PC1-3 cumulative: {s[:3].sum():.1%}")
        print(f"    corr(PC1, r_m) = {r['corr_pc1_rm']:+.4f}    "
              f"corr(PC2, r_m) = {r['corr_pc2_rm']:+.4f}")

    # ---- what is PC2, structurally? --------------------------------------
    print("\nPC2 LOADINGS (is there an interpretable second factor?)\n" + "=" * 74)
    for r in results:
        l2 = r["pc2_loadings"].sort_values()
        lo = ", ".join(f"{k.replace('USDT','')} {v:+.2f}" for k, v in l2.head(4).items())
        hi = ", ".join(f"{k.replace('USDT','')} {v:+.2f}" for k, v in l2.tail(4).items())
        print(f"  {r['label']}  most negative: {lo}")
        print(f"  {'':<6}  most positive: {hi}")

    # ---- stability across years ------------------------------------------
    print("\nSTABILITY BY YEAR (LONG sample)\n" + "=" * 74)
    yr = longs.copy()
    yr["y"] = pd.to_datetime(yr.index, unit="ms", utc=True).year
    print(f"    {'year':<6} {'steps':>6} {'PC1':>7} {'PC2':>7} {'PC3':>7}")
    per_year = {}
    for y, gg in yr.groupby("y"):
        block = gg.drop(columns="y")
        if len(block) < 100:
            continue
        s, _ = eig_shares(np.corrcoef(block.to_numpy(), rowvar=False))
        per_year[int(y)] = [round(float(x), 4) for x in s[:3]]
        print(f"    {y:<6} {len(block):>6,} {s[0]:>6.1%} {s[1]:>6.1%} {s[2]:>6.1%}")

    # ---- GATE 3 -----------------------------------------------------------
    print("\nGATE 3  (pre-registered)\n" + "=" * 74)
    pc2_all = results[0]["shares"][1]
    pc2_long = results[1]["shares"][1]
    pc2_max = max(pc2_all, pc2_long)
    pc2_ci_hi = max(boots["ALL20"]["pc2"][1], boots["LONG"]["pc2"][1])

    single = pc2_max < PC2_TWO_FACTOR
    gate(single,
         f"PC2 < {PC2_TWO_FACTOR:.0%} in both samples -> single factor sufficient",
         f"ALL20 {pc2_all:.1%}, LONG {pc2_long:.1%}")
    # A point estimate under the bar whose interval crosses it is not a clean call.
    gate(pc2_ci_hi < PC2_TWO_FACTOR,
         f"PC2 upper CI also below {PC2_TWO_FACTOR:.0%} (decision is unambiguous)",
         f"worst upper bound {pc2_ci_hi:.1%}")

    c_min = min(r["corr_pc1_rm"] for r in results)
    gate(c_min >= CORR_PC1_MIN,
         f"corr(PC1, r_m) >= {CORR_PC1_MIN} in both samples -> D1 captures the "
         f"dominant factor", f"worst {c_min:.4f}")

    agree = (pc2_all < PC2_TWO_FACTOR) == (pc2_long < PC2_TWO_FACTOR)
    gate(agree, "both samples reach the same verdict")

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"GATE 3 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        print("This gate changes the target's functional form. Resolve, do not defer.")
        return 1

    manifest = {
        "step": 3,
        "pre_registered": {"pc2_two_factor": PC2_TWO_FACTOR,
                           "corr_pc1_min": CORR_PC1_MIN,
                           "expected_pc1": [0.70, 0.85], "expected_pc2": [0.05, 0.12]},
        "samples": {
            r["label"]: {
                "n_coins": r["n_coins"], "n_steps": r["n_steps"],
                "pc1": round(float(r["shares"][0]), 4),
                "pc2": round(float(r["shares"][1]), 4),
                "pc3": round(float(r["shares"][2]), 4),
                "pc2_ci": [round(x, 4) for x in boots[r["label"]]["pc2"]],
                "corr_pc1_rm": round(r["corr_pc1_rm"], 4),
            } for r in results
        },
        "long_members": long_cols,
        "per_year_shares_long": per_year,
        "verdict": "single-factor" if single else "two-factor",
        "bootstrap_draws": args.boot,
    }
    out = DATA_DIR / "grid" / "factor_structure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("GATE 3 PASSED -- SINGLE-FACTOR neutralisation confirmed.")
    print("The target keeps its specified form; beta stays a scalar.")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
