"""Semi-beta gate on the target. TARGET_DESIGN.md §9 item 8.

WHAT GATE 9B ACTUALLY TESTED, AND WHAT IT DID NOT
--------------------------------------------------
Gate 9B split the sample by market regime and checked that corr(target, beta)
does not flip sign between up and down markets. That catches a target whose
LINEAR beta removal is incomplete.

It cannot catch asymmetric loading. A coin can have beta 1.0 on average while
loading 1.4 in falling markets and 0.6 in rising ones. Subtracting a single
beta times the market leaves a residual that is systematically negative when
the market falls and positive when it rises -- and its FULL-SAMPLE correlation
with r_m is near zero, because the two halves cancel. Every linear test passes.
The strategy still carries downside market exposure, which is the exposure that
matters.

This is the gap §9 item 8 recorded. It is closed here.

THE TEST
--------
Semi-betas in the sense of Bollerslev, Patton and Quaedvlieg (2022): estimate
the target's loading on the market SEPARATELY on the two half-samples.

    beta-  =  slope of target on r_m  over bars where r_m < 0
    beta+  =  slope of target on r_m  over bars where r_m > 0

Under a genuinely neutral target both are zero AND their difference is zero.
The difference is the quantity Gate 9B is blind to, so it is the headline here.

SCALE-FREE, FOR THE REASON GATE 9A ALREADY ESTABLISHED
-------------------------------------------------------
A |slope| threshold is not usable on this target. The target is divided by
sigma_eps, so slope = corr x std(target)/std(r_m) ~ corr x 53: a slope bar of
0.05 implies |corr| < 0.00094, tighter than the standard error of a correlation
at this sample size. A perfectly neutral target would fail it most of the time.
Gate 9A superseded its own pre-registered slope bar for exactly this reason and
replaced it with a correlation test against the sampling error. The same
correction applies here, and is applied for the same reason -- not because a
slope test happened to fail.

So the gates are on CORRELATIONS, judged against their own standard errors:

    9C-1  |corr-|  and  |corr+|  each within 1.96 SE of zero
    9C-2  the DIFFERENCE corr- minus corr+ within 1.96 SE of zero
    9C-3  the same difference, per coin, Bonferroni-corrected across 20 coins

TRAIN ONLY. The holdouts are not read.
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

Z = 1.96
failures: list[str] = []


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def _z_two_sided(alpha: float) -> float:
    """Two-sided normal critical value, by bisection -- no scipy dependency."""
    from math import erf, sqrt
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # P(|Z| > mid) = 2 * (1 - Phi(mid))
        p = 1.0 - erf(mid / sqrt(2.0))
        if p > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def corr_se(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Pearson correlation, its standard error, and n."""
    n = len(x)
    if n < 30:
        return np.nan, np.nan, n
    r = float(np.corrcoef(x, y)[0, 1])
    return r, float(np.sqrt(max(1e-12, (1 - r ** 2)) / (n - 2))), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="grid/gate_semibeta.json")
    args = ap.parse_args()

    t = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet")
    t = t[t["t_obs"] < splits.HOLDOUT1_START_MS]
    t = t.dropna(subset=["target", "fwd_rm", "fwd_ret", "beta", "sigma_eps"])
    splits.assert_train_only(t.assign(bar_close_time=t["t_obs"]))

    print("GATE 9C -- SEMI-BETA ON THE TARGET  (TARGET_DESIGN.md §9 item 8)")
    print(f"  {len(t):,} train rows, "
          f"{t['t_obs'].nunique():,} bars, {t['symbol'].nunique()} coins")
    print(f"  closes the gap Gate 9B could not see: ASYMMETRIC loading\n")

    dn = t[t["fwd_rm"] < 0]
    up = t[t["fwd_rm"] > 0]
    print(f"  down-market rows {len(dn):,} ({len(dn)/len(t):.1%}), "
          f"up-market rows {len(up):,} ({len(up)/len(t):.1%})")

    r_dn, se_dn, n_dn = corr_se(dn["target"].to_numpy(), dn["fwd_rm"].to_numpy())
    r_up, se_up, n_up = corr_se(up["target"].to_numpy(), up["fwd_rm"].to_numpy())

    # The same slope numbers, reported but NOT gated -- see the module note.
    def slope(d):
        x, y = d["fwd_rm"].to_numpy(), d["target"].to_numpy()
        return float(np.polyfit(x, y, 1)[0])

    print(f"\n  beta-  (r_m < 0): corr {r_dn:+.5f}  SE {se_dn:.5f}  "
          f"t {r_dn/se_dn:+.2f}   [slope {slope(dn):+.3f}]")
    print(f"  beta+  (r_m > 0): corr {r_up:+.5f}  SE {se_up:.5f}  "
          f"t {r_up/se_up:+.2f}   [slope {slope(up):+.3f}]")

    print("\nGATE 9C-1 -- each half-sample is neutral on its own")
    print("=" * 74)
    gate(abs(r_dn) < Z * se_dn, "down-market loading not significant",
         f"corr {r_dn:+.5f}, t {r_dn/se_dn:+.2f}")
    gate(abs(r_up) < Z * se_up, "up-market loading not significant",
         f"corr {r_up:+.5f}, t {r_up/se_up:+.2f}")

    print("\nGATE 9C-2 -- THE ASYMMETRY ITSELF (what 9B is blind to)")
    print("=" * 74)
    diff = r_dn - r_up
    se_d = float(np.sqrt(se_dn ** 2 + se_up ** 2))
    gate(abs(diff) < Z * se_d,
         "corr(down) - corr(up) not significant",
         f"diff {diff:+.5f}, SE {se_d:.5f}, t {diff/se_d:+.2f}")

    print("\nGATE 9C-3 -- per coin, Bonferroni-corrected")
    print("=" * 74)
    rows = []
    for sym, d in t.groupby("symbol"):
        a, b = d[d["fwd_rm"] < 0], d[d["fwd_rm"] > 0]
        ra, sa, na = corr_se(a["target"].to_numpy(), a["fwd_rm"].to_numpy())
        rb, sb, nb = corr_se(b["target"].to_numpy(), b["fwd_rm"].to_numpy())
        if not (np.isfinite(ra) and np.isfinite(rb)):
            continue
        s = float(np.sqrt(sa ** 2 + sb ** 2))
        rows.append({"symbol": sym, "corr_dn": ra, "corr_up": rb,
                     "diff": ra - rb, "se": s, "t": (ra - rb) / s,
                     "n_dn": na, "n_up": nb})
    P = pd.DataFrame(rows).set_index("symbol").sort_values("t")
    # Bonferroni across the coins actually tested.
    zb = _z_two_sided(0.05 / len(P))
    print(f"  {len(P)} coins, Bonferroni z = {zb:.3f} (alpha 0.05)")
    print(f"\n  {'symbol':<10} {'corr-':>9} {'corr+':>9} {'diff':>9} {'t':>7}")
    for s, r in pd.concat([P.head(4), P.tail(4)]).iterrows():
        print(f"  {s:<10} {r['corr_dn']:>+9.4f} {r['corr_up']:>+9.4f} "
              f"{r['diff']:>+9.4f} {r['t']:>+7.2f}")
    worst = P["t"].abs().max()
    bad = P[P["t"].abs() >= zb]
    gate(len(bad) == 0,
         f"no coin shows significant asymmetry after Bonferroni",
         f"largest |t| {worst:.2f} vs {zb:.2f}"
         + (f"; offenders {list(bad.index)}" if len(bad) else ""))

    # ================= 9C-4: what the failure actually is =================
    print("\nGATE 9C-4 -- diagnosis: convexity, not directional beta")
    print("=" * 74)
    am = np.abs(t["fwd_rm"].to_numpy())

    def ct(a, b):
        r = float(np.corrcoef(a, b)[0, 1])
        n = len(a)
        return r, r * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - r * r))

    r_lin, t_lin = ct(t["target"].to_numpy(), t["fwd_rm"].to_numpy())
    r_abs, t_abs = ct(t["target"].to_numpy(), am)
    print(f"  corr(target, r_m  ) {r_lin:+.5f}  t {t_lin:+7.2f}   <- LINEAR, fine")
    print(f"  corr(target, |r_m|) {r_abs:+.5f}  t {t_abs:+7.2f}   <- the actual defect")
    print("  The target is systematically NEGATIVE when the market moves hard in")
    print("  EITHER direction. A single beta per coin cannot represent a loading")
    print("  that depends on |r_m|, so the residual keeps one.")

    dm = t["target"] - t.groupby("t_obs")["target"].transform("mean")
    r_dm, _ = ct(dm.to_numpy(), am)
    print(f"\n  corr(within-bar demeaned target, |r_m|) {r_dm:+.5f}")
    print("  [TAUTOLOGY, NOT EVIDENCE] |r_m| is constant within a bar and the")
    print("  demeaned target sums to zero within a bar, so this is exactly 0 by")
    print("  construction. It shows the COMMON part cancels; it says nothing")
    print("  about whether coins DIFFER in convexity. They do:")

    per = t.groupby("symbol").apply(
        lambda d: ct(d["target"].to_numpy(), np.abs(d["fwd_rm"].to_numpy()))[0],
        include_groups=False)
    bmean = t.groupby("symbol")["beta"].mean()
    r_cb, t_cb = ct(per.reindex(bmean.index).to_numpy(), bmean.to_numpy())
    print(f"    per-coin corr(target,|r_m|): mean {per.mean():+.4f}  "
          f"std {per.std():.4f}")
    print(f"    range {per.min():+.4f} ({per.idxmin()}) .. "
          f"{per.max():+.4f} ({per.idxmax()})")
    print(f"    corr(per-coin convexity, mean beta) {r_cb:+.3f}  t {t_cb:+.2f}  "
          f"(n={len(bmean)}, underpowered)")
    print("\n  CONSEQUENCE. A cross-sectional ranking objective consumes only")
    print("  within-bar ORDER, so the common convexity cannot reach it. What can")
    print("  reach it is the DISPERSION above: a book that sorts on any")
    print("  characteristic correlated with coin-level convexity inherits the")
    print("  exposure. That is a portfolio-level risk, measurable only once a")
    print("  book exists, and it is registered as such rather than waved away.")

    print("\n" + "=" * 74)
    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "gate": "9C_semibeta", "train_only": True,
        "n_rows": int(len(t)), "n_bars": int(t["t_obs"].nunique()),
        "corr_down": r_dn, "se_down": se_dn, "n_down": n_dn,
        "corr_up": r_up, "se_up": se_up, "n_up": n_up,
        "difference": diff, "se_difference": se_d, "t_difference": diff / se_d,
        "bonferroni_z": zb,
        "diagnosis": {
            "corr_target_rm": r_lin, "t_target_rm": t_lin,
            "corr_target_abs_rm": r_abs, "t_target_abs_rm": t_abs,
            "corr_demeaned_abs_rm": r_dm,
            "demeaned_is_tautological": True,
            "per_coin_convexity_mean": float(per.mean()),
            "per_coin_convexity_std": float(per.std()),
            "verdict": "convexity (|r_m| loading), not directional beta; "
                       "common part cannot reach a ranking objective, "
                       "cross-sectional dispersion is a portfolio-level risk",
        },
        "per_coin": {s: {k: float(v) for k, v in r.items()}
                     for s, r in P.iterrows()},
        "failures": failures,
    }, indent=2), encoding="utf-8")

    if failures:
        print("FAILED: " + "; ".join(failures))
        print(f"FAILED ({len(failures)}) -- and the failure is REAL, but it is")
        print("not the failure the gate names. The target is LINEARLY neutral")
        print(f"(corr with r_m {r_lin:+.5f}, t {t_lin:+.2f}). What it carries is")
        print(f"CONVEXITY: corr with |r_m| {r_abs:+.5f}, t {t_abs:+.2f}.")
        print("")
        print("The design's claim must be stated more narrowly than it was:")
        print("neutral to the LINEAR market factor, not to market convexity.")
        print("The thresholds are NOT relaxed to make this pass -- the gate")
        print("stands as failed and the claim is what changes.")
    else:
        print("The target shows no asymmetric market loading. Gate 9B's blind "
              "spot is closed.")
    print(f"Wrote {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
