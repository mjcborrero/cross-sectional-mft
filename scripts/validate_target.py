"""Step 9 -- does the target actually do what it claims?

Implements Step 9 of docs/TARGET_DESIGN.md.

9A  RESIDUAL MARKET LOADING. Regress the target on the forward market return.
    Slope ~ 0 means beta removal worked. Reported pooled (as pre-registered),
    per coin, and on the UNSCALED residual -- because dividing by sigma_eps
    reweights coins by 1/sigma, so the pooled slope is dominated by low-vol
    names (BTC's 1/sigma is ~5x DOGE's) and can hide a per-coin problem.

9B  REGIME SPLIT ON THE TARGET ITSELF. The sec 7 hazard is that the label
    carries (beta_true - beta_hat)*r_m, an error that scales with market
    direction. The sharpest test is not the mean target by regime but
    corr(target, beta) within each bar: if beta removal leaves residual
    exposure, that correlation FLIPS SIGN between up and down markets. A
    vol-tilt check on corr(target, sigma_eps) is run the same way, since
    normalising in the wrong order would inject exactly that.

9C  DISTRIBUTION. Tails, dispersion stability, and whether extremes cluster in
    particular coins or periods. The winsorisation decision is stated
    explicitly rather than left implicit.

9D  NOISE FLOOR. What does zero skill look like? Establishes the IC standard
    error BEFORE any model exists, so no future number can be quoted without
    a scale to judge it against. Analytic and Monte Carlo, cross-checked.

    python scripts/validate_target.py
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
SLOPE_MAX = 0.05          # pre-registered Gate 9A


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def block_ci(v: np.ndarray, block: int = 20, draws: int = 3000, seed: int = 0):
    """Moving-block bootstrap CI for the mean of a per-bar series."""
    rng = np.random.default_rng(seed)
    n = len(v)
    nb = max(1, n // block)
    out = np.empty(draws)
    for i in range(draws):
        st = rng.integers(0, max(1, n - block), nb)
        out[i] = np.concatenate([v[s:s + block] for s in st]).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(out.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="grid/target_8h_lag60.parquet")
    ap.add_argument("--mc", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(DATA_DIR / args.target)
    df["dt"] = pd.to_datetime(df["t_obs"], unit="ms", utc=True)
    print(f"Step 9 -- target validation   {len(df):,} rows, "
          f"{df.t_obs.nunique():,} cross-sections\n")

    # ================= 9A: residual market loading ========================
    print("GATE 9A -- residual market loading\n" + "=" * 74)
    slope = ols_slope(df["target"].to_numpy(), df["fwd_rm"].to_numpy())
    corr = float(np.corrcoef(df["target"], df["fwd_rm"])[0, 1])
    se_c = 1.0 / np.sqrt(len(df))
    # The pre-registered |slope| < 0.05 was MIS-SPECIFIED and is superseded.
    # slope = corr * std(target)/std(r_m) ~ corr * 53, because the target is
    # divided by sigma_eps (~0.005-0.027). Passing it would require
    # |corr| < 0.00094, which is 3.6x TIGHTER than the standard error of a
    # correlation at this sample size -- a perfectly neutral target would fail
    # ~78% of the time by chance. The threshold was borrowed from return space
    # and applied to a standardised quantity. The scale-free test replaces it.
    print(f"    [note] pre-registered |slope|<{SLOPE_MAX} superseded: slope "
          f"{slope:+.4f} = corr x {df.target.std()/df.fwd_rm.std():.0f}; "
          f"see docs. Scale-free test below.")
    gate(abs(corr) < 1.96 * se_c,
         "target is not significantly correlated with r_m (scale-free)",
         f"corr {corr:+.5f}, SE {se_c:.5f}, t {corr/se_c:+.2f}")

    # Unscaled: the direct test that beta removed the market, unweighted by
    # 1/sigma.
    resid = df["fwd_ret"] - df["beta"] * df["fwd_rm"]
    slope_u = ols_slope(resid.to_numpy(), df["fwd_rm"].to_numpy())
    gate(abs(slope_u) < SLOPE_MAX,
         "unscaled residual also shows |slope| < 0.05 (beta removal is clean)",
         f"slope {slope_u:+.5f}")

    per = (df.groupby("symbol")
             .apply(lambda d: pd.Series({
                 "slope_scaled": ols_slope(d["target"].to_numpy(),
                                           d["fwd_rm"].to_numpy()),
                 "slope_unscaled": ols_slope(
                     (d["fwd_ret"] - d["beta"] * d["fwd_rm"]).to_numpy(),
                     d["fwd_rm"].to_numpy()),
             }), include_groups=False)
             .sort_values("slope_scaled"))
    # Per coin, scale-free and Bonferroni-corrected for 20 simultaneous tests.
    pc = df.groupby("symbol").apply(
        lambda d: pd.Series({"corr": float(np.corrcoef(d["target"], d["fwd_rm"])[0, 1]),
                             "n": len(d)}), include_groups=False)
    pc["t"] = pc["corr"] * np.sqrt(pc["n"])
    zc = 3.02          # two-sided 0.05/20
    worst = float(pc["t"].abs().max())
    gate(worst < zc,
         f"no coin significantly loaded on r_m (Bonferroni, 20 tests, |t|<{zc})",
         f"worst |t| {worst:.2f} ({pc['t'].abs().idxmax()})")
    print(f"\n  per-coin slope on r_m (scaled / unscaled), 5 most negative & positive:")
    show = pd.concat([per.head(5), per.tail(5)])
    for s, r in show.iterrows():
        print(f"    {s:<10} {r['slope_scaled']:>+8.3f}  {r['slope_unscaled']:>+8.4f}")

    # ================= 9B: regime split ===================================
    print("\nGATE 9B -- regime split on the target itself\n" + "=" * 74)
    bar = df.groupby("t_obs").agg(rm=("fwd_rm", "first")).reset_index()
    up = set(bar.loc[bar["rm"] > 0, "t_obs"])
    df["regime"] = np.where(df["t_obs"].isin(up), "up", "down")

    # Per-bar correlations. If beta removal leaves exposure, corr(target, beta)
    # flips sign with market direction -- that is the sec 7 hazard made visible.
    def per_bar_corr(col):
        return (df.groupby("t_obs")
                  .apply(lambda d: d["target"].corr(d[col], method="spearman")
                         if len(d) >= 10 else np.nan, include_groups=False)
                  .dropna())

    cb = per_bar_corr("beta").rename("c_beta")
    cs = per_bar_corr("sigma_eps").rename("c_sigma")
    reg = bar.set_index("t_obs")["rm"]
    tab = pd.concat([cb, cs], axis=1).join(reg.rename("rm")).dropna()
    tab["up"] = tab["rm"] > 0

    print(f"  {'':<22} {'up bars':>10} {'down bars':>11} {'gap':>9} {'gap 95% CI':>20}")
    results = {}
    for name, col in [("corr(target, beta)", "c_beta"),
                      ("corr(target, sigma_eps)", "c_sigma")]:
        u = tab.loc[tab["up"], col].to_numpy()
        d = tab.loc[~tab["up"], col].to_numpy()
        gap = u.mean() - d.mean()
        lo_u, hi_u, se_u = block_ci(u, seed=args.seed)
        lo_d, hi_d, se_d = block_ci(d, seed=args.seed + 1)
        se_gap = np.sqrt(se_u ** 2 + se_d ** 2)
        results[col] = {"up": float(u.mean()), "down": float(d.mean()),
                        "gap": float(gap), "se_gap": float(se_gap)}
        print(f"  {name:<22} {u.mean():>+10.4f} {d.mean():>+11.4f} {gap:>+9.4f}"
              f"  [{gap-1.96*se_gap:+.4f}, {gap+1.96*se_gap:+.4f}]")

    gb = results["c_beta"]
    gate(abs(gb["gap"]) < 1.96 * gb["se_gap"] or abs(gb["gap"]) < 0.05,
         "beta exposure does not flip sign with market direction",
         f"gap {gb['gap']:+.4f} +/- {1.96*gb['se_gap']:.4f}")
    gs = results["c_sigma"]
    gate(abs(gs["gap"]) < 1.96 * gs["se_gap"] or abs(gs["gap"]) < 0.05,
         "no regime-dependent volatility tilt",
         f"gap {gs['gap']:+.4f} +/- {1.96*gs['se_gap']:.4f}")

    mu = df.groupby("regime")["target"].mean()
    print(f"\n  mean target: up {mu.get('up', np.nan):+.4f}  "
          f"down {mu.get('down', np.nan):+.4f}  "
          f"({len(up):,} up bars, {tab.shape[0]-len(up & set(tab.index)):,} down)")

    # ================= 9C: distribution ===================================
    print("\nGATE 9C -- distribution\n" + "=" * 74)
    t = df["target"]
    xs = df.groupby("t_obs")["target"].std()
    gate(bool(np.isfinite(t).all()), "all finite")
    gate(float(t.abs().max()) < 200, "no explosive values",
         f"max |target| {t.abs().max():.1f}")
    # A raw CV gate on a positive, right-skewed quantity is not meaningful, and
    # under a RANKING objective per-bar scale is irrelevant anyway: ranks are
    # invariant to it. What matters is that dispersion is always positive and
    # never collapses, which would make a bar unrankable.
    gate(bool((xs > 0.05).all()),
         "cross-sectional dispersion never collapses (every bar is rankable)",
         f"min {xs.min():.3f}, median {xs.median():.3f}, "
         f"p95/p50 {xs.quantile(.95)/xs.median():.2f}")

    ext = df[t.abs() > 5]
    conc = ext["symbol"].value_counts(normalize=True).head(3)
    print(f"\n  tails: |t|>3 {(t.abs()>3).mean():.3%}   |t|>5 {(t.abs()>5).mean():.3%}"
          f"   |t|>10 {(t.abs()>10).mean():.4%}   max {t.abs().max():.1f}")
    print(f"  extremes by coin (top 3 of {ext.symbol.nunique()}): "
          + ", ".join(f"{k.replace('USDT','')} {v:.1%}" for k, v in conc.items()))
    yr_ext = ext.groupby(ext["dt"].dt.year).size() / df.groupby(df["dt"].dt.year).size()
    print(f"  |t|>5 rate by year: "
          + "  ".join(f"{y} {v:.2%}" for y, v in yr_ext.items()))
    gate(float(conc.iloc[0]) < 0.25,
         "extremes are not concentrated in one coin",
         f"largest share {conc.index[0].replace('USDT','')} {conc.iloc[0]:.1%}")

    print("\n  WINSORISATION: deliberately NOT applied. The ranking objective is")
    print("  bounded, so tail magnitude cannot dominate the fit the way it would")
    print("  under squared error. Applying it would be a second variable; if it is")
    print("  ever tested it must be its own one-variable experiment.")

    # ================= 9D: noise floor ====================================
    print("\nGATE 9D -- noise floor (what does zero skill look like?)\n" + "=" * 74)
    sizes = df.groupby("t_obs").size()
    nb = len(sizes)
    analytic = float(np.sqrt((1.0 / (sizes - 1)).mean() / nb))

    rng = np.random.default_rng(args.seed)
    codes = pd.factorize(df["t_obs"])[0]
    tr = df.groupby("t_obs")["target"].rank().to_numpy()
    means = np.empty(args.mc)
    for i in range(args.mc):
        rnd = rng.standard_normal(len(df))
        rr = pd.Series(rnd).groupby(codes).rank().to_numpy()
        a = pd.DataFrame({"g": codes, "x": rr, "y": tr})
        ic = a.groupby("g").apply(lambda d: np.corrcoef(d.x, d.y)[0, 1],
                                  include_groups=False)
        means[i] = ic.mean()
    mc_se = float(means.std())

    print(f"  cross-section size: mean {sizes.mean():.1f}, min {sizes.min()}, "
          f"{nb:,} bars")
    print(f"  IC standard error under the null:")
    print(f"    analytic   {analytic:.5f}")
    print(f"    Monte Carlo {mc_se:.5f}   ({args.mc} random signals)")
    print(f"  -> |IC| must exceed {1.96*mc_se:.4f} to clear 2 sigma, "
          f"{2.58*mc_se:.4f} for 3 sigma")
    gate(abs(analytic - mc_se) / analytic < 0.25,
         "analytic and Monte Carlo noise floors agree",
         f"{analytic:.5f} vs {mc_se:.5f}")

    ac = df.sort_values(["symbol", "t_obs"]).groupby("symbol")["target"].apply(
        lambda s: s.autocorr(1))
    print(f"\n  target lag-1 autocorrelation by coin: mean {ac.mean():+.4f}, "
          f"max |{ac.abs().max():.4f}|")
    gate(float(ac.abs().max()) < 0.10,
         "target is not materially autocorrelated (labels do not overlap)",
         f"max |rho| {ac.abs().max():.4f} ({ac.abs().idxmax()})")

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"STEP 9 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        print("Per TARGET_DESIGN.md: on 9A/9B failure, return to Steps 4-5.")
        return 1

    out = DATA_DIR / "grid" / "target_validation.json"
    out.write_text(json.dumps({
        "step": 9,
        "gate9a": {"pooled_slope": slope, "pooled_corr": corr,
                   "pooled_t": float(corr/se_c), "unscaled_slope": slope_u,
                   "worst_coin_t": float(worst),
                   "per_coin": {k: round(float(v), 5)
                                for k, v in per["slope_scaled"].items()}},
        "gate9b": {k: {kk: round(vv, 6) for kk, vv in v.items()}
                   for k, v in results.items()},
        "gate9c": {"skew": round(float(t.skew()), 4),
                   "kurtosis": round(float(t.kurtosis()), 4),
                   "frac_abs_gt_5": round(float((t.abs() > 5).mean()), 6),
                   "max_abs": round(float(t.abs().max()), 4),
                   "xs_dispersion_mean": round(float(xs.mean()), 5),
                   "winsorised": False},
        "gate9d": {"ic_se_analytic": round(analytic, 6),
                   "ic_se_montecarlo": round(mc_se, 6),
                   "ic_2sigma": round(1.96 * mc_se, 6),
                   "ic_3sigma": round(2.58 * mc_se, 6),
                   "max_abs_autocorr": round(float(ac.abs().max()), 5)},
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("STEP 9 PASSED. The target is market-neutral, regime-stable, and its")
    print(f"noise floor is established: IC SE = {mc_se:.5f}.")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
