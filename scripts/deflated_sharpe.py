"""Probabilistic and Deflated Sharpe Ratio for the frozen book.

Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio". Two questions a
raw Sharpe does not answer:

  PSR   Given the observed Sharpe, its sample size, skew and kurtosis, what is
        the probability that the TRUE Sharpe exceeds a benchmark SR*?

            PSR(SR*) = Phi( (SR - SR*) sqrt(n-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2) )

        with SR per period, g3 skew, g4 kurtosis (Pearson; normal = 3).
        Fat tails and negative skew widen the denominator and lower PSR.

  DSR   PSR evaluated not against zero but against the Sharpe you would EXPECT
        the best of N random trials to show. If N configurations were tried
        and the best reported, the benchmark is the expected maximum:

            SR0 = sqrt(V) [ (1-g) Z(1 - 1/N) + g Z(1 - 1/(N e)) ]

        g = Euler-Mascheroni, V = variance of the trial Sharpes, Z = Phi^-1.
        DSR = PSR(SR0) is the probability the reported Sharpe is not just the
        luckiest of N draws.

WHAT COUNTS AS A TRIAL HERE, AND WHAT DOES NOT
-----------------------------------------------
Trials are configurations of THIS strategy among which a Sharpe-based choice
was made before the spec was frozen:

  fee_ceiling.json   book type x smoothing (lam, band) sweep   -> book C, no smoothing
  fix_hedge.json     hedge variants                             -> rolling-90

Their Sharpes are read from the result files, so N and V are measured, not
assumed.

NOT deflated, stated plainly: feature selection (seven staged filters over
a larger candidate set) and hyperparameter tuning (9 configurations) were selected on
IC, not Sharpe. They ARE selection and they DID shape the reported number,
but there are no Sharpes for them to put into V. The tuning configurations
are added to N at the observed V as a partial correction. The feature search
is not. The DSR reported here is therefore an UPPER BOUND on the true one.

The trials are also highly correlated (variations of one book on one panel),
which means N overstates the effective number of independent trials -- that
pushes SR0 UP and DSR DOWN, i.e. it is the conservative direction.

THE HOLDOUTS GET PSR, NOT DSR
------------------------------
Each holdout was a single, pre-registered, spend-once test. There was no
selection among trials, so the benchmark is SR* = 0 and the statistic is
PSR(0): the probability the true out-of-sample Sharpe is positive, given
what was observed. The holdout return series are recomputed here and their
Sharpes asserted against the recorded result files, so the series being
evaluated is the one that was reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits, strategy
from mft.paths import DATA_DIR
from scripts.run_strategy import attach_funding, cache_path, load_panel
from scripts.stage7_importance import LGB_PARAMS, SEEDS

BY = strategy.BARS_PER_YEAR
EULER = 0.5772156649
CAP = int(pd.Timestamp("2026-07-31", tz="UTC").timestamp() * 1000)


def psr(ret: np.ndarray, sr_star: float = 0.0) -> dict:
    """PSR of a per-period return series against per-period benchmark sr_star."""
    r = np.asarray(ret, float)
    r = r[np.isfinite(r)]
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = float(sps.skew(r))
    g4 = float(sps.kurtosis(r, fisher=False))          # Pearson, normal = 3
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    z = (sr - sr_star) * np.sqrt(n - 1) / denom
    return {"n": n, "sr_bar": float(sr), "sr_ann": float(sr * np.sqrt(BY)),
            "skew": g3, "kurt": g4, "z": float(z), "psr": float(sps.norm.cdf(z))}


def expected_max_sr(var_trials: float, n_trials: int) -> float:
    """SR0: expected maximum Sharpe of n_trials draws with variance var_trials."""
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trials) * ((1 - EULER) * z1 + EULER * z2))


def min_trl(ret: np.ndarray, sr_star: float, conf: float = 0.95) -> float:
    """Minimum track record length (periods) to reject SR <= sr_star at conf."""
    r = np.asarray(ret, float)
    sr = r.mean() / r.std(ddof=1)
    g3 = float(sps.skew(r))
    g4 = float(sps.kurtosis(r, fisher=False))
    if sr <= sr_star:
        return np.inf
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr * sr)
                 * (sps.norm.ppf(conf) / (sr - sr_star)) ** 2)


def collect_trials() -> tuple[list[float], dict]:
    """Every Sharpe-evaluated configuration the frozen spec was chosen from."""
    F = DATA_DIR / "features"
    trials, src = [], {}
    fc = json.loads((F / "fee_ceiling.json").read_text())
    s = [v["sharpe"] for v in fc["books"].values()]
    s += [g["sharpe"] for g in fc["grid"]]
    s += [g["sharpe"] for g in fc.get("combined_grid", [])]
    src["fee_ceiling (book x smoothing)"] = len(s); trials += s
    fh = json.loads((F / "fix_hedge.json").read_text())
    s = [v["train"]["sharpe"] for v in fh["variants"]]
    src["fix_hedge (hedge variants)"] = len(s); trials += s
    tn = json.loads((F / "tune.json").read_text())
    n_tune = len(tn["grid"])
    src["tune (IC-selected; counted in N, no Sharpe for V)"] = n_tune
    return trials, {"sources": src, "n_with_sharpe": len(trials),
                    "n_tune_extra": n_tune}


def holdout_returns(lo: int, hi: int | None) -> pd.Series:
    """Replicates run_strategy's holdout branch: one fit on all data before
    lo, applied forward, hedge history and book state carried across."""
    import lightgbm as lgb
    Xtr, Ttr, cols, _ = load_panel(lo, lag=60)
    Xho, Tho, _, _ = load_panel(hi, lower_ms=lo, lag=60)
    Rtr = Xtr.groupby(level="t_obs").rank(pct=True)
    Rho = Xho.groupby(level="t_obs").rank(pct=True)
    Ytr = Ttr["target"].groupby(level="t_obs").rank(pct=True).to_numpy()
    acc = np.zeros(len(Xho))
    for sd in SEEDS:
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
            Rtr.to_numpy("float32"), Ytr)
        p = m.predict(Rho.to_numpy("float32"))
        acc += (p - p.mean()) / (p.std() or 1.0)
    P = attach_funding(Tho.assign(score=acc / len(SEEDS)))
    Ptr = pd.read_parquet(cache_path(60))
    rtr = strategy.run(Ptr)
    seed_hist = (rtr.ret.to_numpy(),
                 Ptr.groupby(level="t_obs")["fwd_rm"].first()
                 .reindex(rtr.ret.index).to_numpy())
    iw = rtr.weights.iloc[-1].reindex(
        sorted(P.index.get_level_values("symbol").unique())).fillna(0.0)
    return strategy.run(P, seed=seed_hist, init_weights=iw.to_numpy()).ret


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/deflated_sharpe.json")
    a = ap.parse_args()
    F = DATA_DIR / "features"
    out = {}

    # ---- TRAIN: DSR --------------------------------------------------
    r_tr = strategy.run(pd.read_parquet(cache_path(60))).ret.to_numpy()
    rec = json.loads((F / "strategy_train.json").read_text())["metrics"]["sharpe"]
    base = psr(r_tr, 0.0)
    assert abs(base["sr_ann"] - rec) < 0.02, (base["sr_ann"], rec)
    trials, tinfo = collect_trials()
    tr_bar = np.array(trials) / np.sqrt(BY)              # to per-bar units
    V = float(tr_bar.var(ddof=1))
    N = tinfo["n_with_sharpe"] + tinfo["n_tune_extra"]
    sr0 = expected_max_sr(V, N)
    dsr = psr(r_tr, sr0)

    print("TRAIN -- Deflated Sharpe Ratio")
    print("=" * 70)
    print(f"  observed Sharpe        {base['sr_ann']:+.3f} ann  "
          f"({base['sr_bar']:+.5f} per bar, n={base['n']:,})")
    print(f"  skew {base['skew']:+.3f}   kurtosis {base['kurt']:.2f}  "
          f"(normal = 3)")
    print(f"  PSR vs 0               {base['psr']:.6f}   (z {base['z']:+.2f})")
    print()
    print("  trials the spec was selected from:")
    for k, v in tinfo["sources"].items():
        print(f"    {k:<52} {v:>3}")
    print(f"    {'N used':<52} {N:>3}")
    print(f"  trial Sharpes: mean {np.mean(trials):+.3f}  sd "
          f"{np.std(trials, ddof=1):.3f}  range {min(trials):+.2f}..{max(trials):+.2f}  (ann)")
    print(f"  expected max of N     SR0 = {sr0 * np.sqrt(BY):+.3f} ann")
    print(f"  DSR = PSR(SR0)         {dsr['psr']:.6f}   (z {dsr['z']:+.2f})")
    print(f"  MinTRL to beat SR0 at 95%: {min_trl(r_tr, sr0):,.0f} bars "
          f"(have {base['n']:,})")
    print()
    print("  NOT deflated: feature selection (7 IC-based stages) -- so this")
    print("  DSR is an upper bound. Trials are correlated, which pushes the")
    print("  other way. Both are stated in the docstring.")
    out["train"] = {"psr0": base, "trials": tinfo, "trial_sharpes_ann": trials,
                    "V_bar": V, "N": N, "sr0_ann": sr0 * np.sqrt(BY),
                    "dsr": dsr, "min_trl_bars": min_trl(r_tr, sr0)}

    # ---- HOLDOUTS: PSR vs 0 --------------------------------------------
    for tag, lo, hi, recf in (("2025", splits.TRUE_HOLDOUT1_MS, splits.TRUE_HOLDOUT2_MS,
                               "strategy_holdout1"),
                              ("2026", splits.TRUE_HOLDOUT2_MS, CAP, "strategy_holdout2")):
        r = holdout_returns(lo, hi).to_numpy()
        rec = json.loads((F / f"{recf}.json").read_text())["metrics"]["sharpe"]
        p = psr(r, 0.0)
        assert abs(p["sr_ann"] - rec) < 0.02, (tag, p["sr_ann"], rec)
        print(f"\n{tag} -- Probabilistic Sharpe Ratio (single pre-registered test)")
        print("=" * 70)
        print(f"  observed Sharpe   {p['sr_ann']:+.3f} ann   n={p['n']:,}   "
              f"(matches {recf}.json: {rec:+.3f})")
        print(f"  skew {p['skew']:+.3f}   kurtosis {p['kurt']:.2f}")
        print(f"  PSR vs 0          {p['psr']:.4f}   (z {p['z']:+.2f})")
        print(f"  -> P(true Sharpe > 0) = {p['psr']:.1%}")
        out[f"h{tag}"] = p

    pth = DATA_DIR / a.out
    pth.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
