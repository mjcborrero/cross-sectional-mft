"""End-to-end audit. Does the result make sense, not does the code run.

The previous incarnation of this project died because parts were assumed to
work rather than checked against what they should produce. Every check here is
written to FALSIFY the result, not to confirm it. A check that cannot fail is
not in this file.

    1  PLACEBO           random scores through the same machinery must give
                         Sharpe ~ 0. If they do not, the BOOK is manufacturing
                         the return and the model is decoration.
    2  P&L RECONSTRUCTION the book return recomputed from prices, independently
                         of the fwd_ret column that the backtest consumed.
    3  TARGET IDENTITY   target == (fwd_ret - beta*fwd_rm) / sigma_eps, exactly.
    4  BOOK INVARIANTS   sum(w) = 0 and sum|w| = 1 every bar, not on average.
    5  PURGE             no training label window may reach into its test block.
    6  HOLDOUT SEAL      the out-of-fold score panel that research reads must
                         be train-only. Building a feature panel over a holdout
                         does NOT spend it; using its outcomes does.
    7  CONCENTRATION     is the result one coin, one month, one trade?
    8  CEILING           perfect-foresight Sharpe, to place 2.86 on a scale.
    9  DOSE-RESPONSE     degrading the score must degrade the Sharpe smoothly;
                         a result that survives heavy noise is not coming from
                         the signal.
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

CACHE = DATA_DIR / "features/oof_book_inputs.parquet"
RNG = np.random.default_rng(11)
fails: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(label)
    return ok


def sharpe_of(P: pd.DataFrame, score: pd.Series) -> tuple[float, float]:
    r = strategy.run(P.assign(score=score))
    m = strategy.metrics(r.ret, turnover=float(r.turnover.mean()))
    return m["sharpe"], m["ann_return"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/audit.json")
    args = ap.parse_args()
    P = pd.read_parquet(CACHE)
    res = {}

    base_s, base_r = sharpe_of(P, P["score"])
    print(f"AUDIT -- baseline Sharpe {base_s:+.2f}, ann {base_r:+.2%}, "
          f"{len(P):,} rows\n")

    # ---- 1. placebo -----------------------------------------------------
    print("1. PLACEBO -- random and shuffled scores through the same machinery")
    print("=" * 74)
    rand, shuf = [], []
    bar = P.index.get_level_values("t_obs").to_numpy()
    for _ in range(20):
        rand.append(sharpe_of(P, pd.Series(RNG.standard_normal(len(P)),
                                           index=P.index))[0])
        sh = P["score"].groupby(level="t_obs").transform(
            lambda s: RNG.permutation(s.to_numpy()))
        shuf.append(sharpe_of(P, sh)[0])
    rand, shuf = np.array(rand), np.array(shuf)
    print(f"  random score   Sharpe {rand.mean():+.3f} +/- {rand.std():.3f} "
          f"(range {rand.min():+.2f} .. {rand.max():+.2f})")
    print(f"  shuffled score Sharpe {shuf.mean():+.3f} +/- {shuf.std():.3f} "
          f"(range {shuf.min():+.2f} .. {shuf.max():+.2f})")
    check(abs(rand.mean()) < 0.5, "random score gives ~0 Sharpe",
          f"{rand.mean():+.3f}")
    check(abs(shuf.mean()) < 0.5, "within-bar shuffled score gives ~0 Sharpe",
          f"{shuf.mean():+.3f}")
    z = (base_s - rand.mean()) / rand.std() if rand.std() > 0 else np.inf
    check(z > 5, "baseline is far outside the placebo distribution",
          f"z = {z:.1f}")
    res["placebo"] = {"random_mean": float(rand.mean()),
                      "random_sd": float(rand.std()),
                      "shuffled_mean": float(shuf.mean()),
                      "shuffled_sd": float(shuf.std()), "z": float(z)}

    # ---- 2. P&L reconstruction ------------------------------------------
    print("\n2. P&L RECONSTRUCTION -- from prices, not from the fwd_ret column")
    print("=" * 74)
    fwd = pd.read_parquet(DATA_DIR / "grid/forward_8h_lag60.parquet") \
        .set_index(["t_obs", "symbol"])
    px = fwd["px_fill"].unstack("symbol").sort_index()
    # bars are 8h apart and the window is [t_fill, t_fill + 8h], so the
    # realised return is next bar's fill price over this bar's.
    rebuilt = (px.shift(-1) / px - 1.0).stack()
    rebuilt.index = rebuilt.index.set_names(["t_obs", "symbol"])
    j = P.index.intersection(rebuilt.index)
    a, b = P.loc[j, "fwd_ret"], rebuilt.loc[j]
    both = a.notna() & b.notna()
    md = float((a[both] - b[both]).abs().max())
    print(f"  compared {int(both.sum()):,} rows")
    check(md < 1e-9, "fwd_ret equals px_fill(t+1)/px_fill(t) - 1",
          f"max|diff| {md:.3e}")
    res["pnl_reconstruction_maxdiff"] = md

    # ---- 3. target identity ---------------------------------------------
    print("\n3. TARGET IDENTITY")
    print("=" * 74)
    T = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet")
    T = T[T["t_obs"] < splits.HOLDOUT1_START_MS].dropna(
        subset=["target", "fwd_ret", "fwd_rm", "beta", "sigma_eps"])
    lhs = T["target"].to_numpy()
    rhs = ((T["fwd_ret"] - T["beta"] * T["fwd_rm"]) / T["sigma_eps"]).to_numpy()
    md3 = float(np.nanmax(np.abs(lhs - rhs)))
    check(md3 < 1e-9, "target == (fwd_ret - beta*fwd_rm)/sigma_eps",
          f"max|diff| {md3:.3e} over {len(T):,} rows")
    res["target_identity_maxdiff"] = md3

    # ---- 4. book invariants ---------------------------------------------
    print("\n4. BOOK INVARIANTS -- every bar, not on average")
    print("=" * 74)
    r = strategy.run(P)
    W = r.weights
    s0 = W.sum(axis=1).abs().max()
    s1 = (W.abs().sum(axis=1) - 1.0).abs().max()
    check(s0 < 1e-9, "sum(w) = 0 in every bar", f"worst {s0:.3e}")
    check(s1 < 1e-9, "sum|w| = 1 in every bar", f"worst {s1:.3e}")
    res["invariants"] = {"max_abs_sum_w": float(s0), "max_gross_dev": float(s1)}

    # ---- 5. purge --------------------------------------------------------
    print("\n5. PURGE -- a training label must not reach into its test block")
    print("=" * 74)
    tall = np.sort(P.index.get_level_values("t_obs").unique())
    horizon_ms = (60 + 8 * 60) * 60 * 1000        # lag + h
    worst_gap, bad = None, 0
    for f in foldmod.make_folds(np.sort(np.unique(
            pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet",
                            columns=["t_obs"])
            .query(f"t_obs < {splits.HOLDOUT1_START_MS}")["t_obs"].unique()))):
        last_train_label_end = int(f.train[-1]) + horizon_ms
        test_start = int(f.test[0])
        gap = test_start - last_train_label_end
        worst_gap = gap if worst_gap is None else min(worst_gap, gap)
        if gap < 0:
            bad += 1
        print(f"    fold {f.k}: last train label ends "
              f"{splits._fmt(last_train_label_end)}, test starts "
              f"{splits._fmt(test_start)}, gap {gap/3.6e6:+.1f}h")
    check(bad == 0, "no training label overlaps its test block",
          f"tightest gap {worst_gap/3.6e6:+.1f}h")
    res["purge_min_gap_hours"] = float(worst_gap / 3.6e6)

    # ---- 6. holdout seal -------------------------------------------------
    # WHAT "SPENT" MEANS, because the naive version of this check has now been
    # wrong twice.
    #
    # A feature panel built over 2026 does NOT spend 2026. Features are
    # deterministic functions of data at or before their own timestamp; nothing
    # was fitted, selected or tuned on them. Building the panel is a
    # PREREQUISITE to evaluating a holdout, so a seal that forbids it forbids
    # the evaluation itself.
    #
    # What spends a holdout is letting its OUTCOMES inform a decision. In this
    # pipeline outcomes enter research through exactly one door: the cached
    # out-of-fold score panel that every downstream analysis reads. THAT is
    # what must stay train-only, and that is what is checked.
    #
    # The feature panels' extent is REPORTED, not gated, so the reader can see
    # how far they run without the check crying wolf.
    print("\n6. HOLDOUT SEAL -- what research reads, not what exists on disk")
    print("=" * 74)
    cache_max = int(P.index.get_level_values("t_obs").max())
    check(cache_max < splits.TRUE_HOLDOUT1_MS,
          "the out-of-fold score panel research reads is TRAIN-ONLY",
          f"ends {splits._fmt(cache_max)}")

    spent = ", ".join(splits.SPENT_HOLDOUTS) or "none"
    print(f"       holdouts spent so far: {spent}")
    extents = {}
    for p in sorted((DATA_DIR / "features").glob("family_*.parquet")):
        d = pd.read_parquet(p, columns=["t_obs"])
        extents[p.name] = int(d["t_obs"].max())
    if extents:
        mx = max(extents.values())
        print(f"       feature panels run to {splits._fmt(mx)} "
              f"({len(extents)} families) -- built deliberately, not a leak")
    res["seal"] = {"cache_end": splits._fmt(cache_max),
                   "spent": list(splits.SPENT_HOLDOUTS),
                   "feature_panel_end": splits._fmt(max(extents.values()))
                   if extents else None}

    # ---- 7. concentration ------------------------------------------------
    print("\n7. CONCENTRATION -- is this one coin or one month?")
    print("=" * 74)
    R = P["fwd_ret"].unstack("symbol").reindex(W.index).reindex(columns=W.columns)
    contrib = (W * R.fillna(0.0)).sum(axis=0).sort_values()
    tot = contrib.sum()
    top = contrib.abs().sort_values(ascending=False)
    share = float(top.iloc[0] / abs(tot)) if tot else np.nan
    print(f"  top contributor {top.index[0]}: {share:.1%} of total gross P&L")
    print("  bottom 3: " + ", ".join(f"{i} {contrib[i]:+.3f}"
                                     for i in contrib.index[:3]))
    print("  top 3:    " + ", ".join(f"{i} {contrib[i]:+.3f}"
                                     for i in contrib.index[-3:]))
    check(share < 0.35, "no single coin dominates the P&L", f"{share:.1%}")
    mo = r.ret.groupby(pd.to_datetime(r.ret.index, unit="ms").to_period("M")).sum()
    mshare = float(mo.abs().max() / abs(mo.sum())) if mo.sum() else np.nan
    print(f"  best month is {mshare:.1%} of total; "
          f"{int((mo > 0).sum())}/{len(mo)} months positive")
    check(mshare < 0.35, "no single month dominates", f"{mshare:.1%}")
    res["concentration"] = {"top_coin": str(top.index[0]),
                            "top_coin_share": share,
                            "best_month_share": mshare,
                            "months_positive": int((mo > 0).sum()),
                            "months": int(len(mo))}

    # ---- 8. ceiling ------------------------------------------------------
    print("\n8. CEILING -- perfect foresight, to place the result on a scale")
    print("=" * 74)
    perfect, _ = sharpe_of(P, P.join(
        T.set_index(["t_obs", "symbol"])["target"].rename("tgt"))["tgt"])
    print(f"  perfect-foresight Sharpe {perfect:+.1f} vs realised {base_s:+.2f}"
          f"  ({base_s/perfect:.1%} of ceiling)")
    check(perfect > base_s * 3, "realised is far below perfect foresight",
          f"ratio {base_s/perfect:.3f}")
    res["ceiling"] = {"perfect": float(perfect), "realised": float(base_s),
                      "fraction": float(base_s / perfect)}

    # ---- 9. dose-response ------------------------------------------------
    print("\n9. DOSE-RESPONSE -- adding noise must degrade the result smoothly")
    print("=" * 74)
    sd = float(P["score"].std())
    dr = []
    for k in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        s = P["score"] + pd.Series(RNG.standard_normal(len(P)) * sd * k,
                                   index=P.index)
        v = sharpe_of(P, s)[0]
        dr.append({"noise_x_sd": k, "sharpe": float(v)})
        print(f"    noise {k:>4.1f} x sd   Sharpe {v:>+6.2f}")
    mono = all(dr[i]["sharpe"] >= dr[i + 1]["sharpe"] - 0.15
               for i in range(len(dr) - 1))
    check(mono, "Sharpe decreases monotonically with noise")
    check(dr[-1]["sharpe"] < base_s * 0.5,
          "heavy noise destroys the result (it comes from the signal)",
          f"{dr[-1]['sharpe']:+.2f} vs {base_s:+.2f}")
    res["dose_response"] = dr

    print("\n" + "=" * 74)
    if fails:
        print(f"AUDIT FAILED ({len(fails)}):")
        for f_ in fails:
            print(f"    {f_}")
    else:
        print("ALL AUDIT CHECKS PASSED.")
        print("This does not prove the strategy works out of sample. It says")
        print("the machinery computes what it claims and the result is not an")
        print("artifact of the book, a leak, or one lucky coin.")
    res["failures"] = fails
    out = DATA_DIR / args.out
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"Wrote {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
