"""Size the book by FORECAST dispersion. Does the quiet/mid edge survive?

THE FINDING THIS TESTS
-----------------------
scripts/ic_by_dispersion.py: the Q5-Q1 spread is +9.11 bps (train) and +9.27
bps (2026) on quiet bars, +9.17 / +10.27 on mid bars, and -14.57 / -35.52 on
violent ones. The edge on quiet bars did not degrade AT ALL out-of-sample; the
book dies entirely in the violent tercile.

scripts/dispersion_predictable.py: trailing dispersion computed strictly before
t_obs forecasts forward dispersion at rank corr 0.61 (train) / 0.38 (2026),
with P(violent | called quiet) 11% / 19% against 33% for a coin flip.

So: stand down when dispersion is forecast to be high.

THREE THINGS THAT WOULD MAKE THIS A LIE, AND HOW EACH IS AVOIDED
-----------------------------------------------------------------
1. LOOK-AHEAD IN THE THRESHOLD. Deciding "top tercile" from full-sample
   quantiles uses the future to set the cut. Percentiles here are EXPANDING --
   bar t is ranked only against bars strictly before it, after a 200-bar
   warm-up. That costs the first 200 bars and buys an honest gate.

2. PICKING THE WINDOW ON 2026. The 3-bar window was best on 2026 and the 9-bar
   on train. Choosing either by its best column is selection on spent data, so
   the whole grid is printed and read for a PLATEAU. Same discipline that
   failed the 30d momentum lookback.

3. PRETENDING TURNOVER IS FREE. It is, by this project's stated design
   assumption of zero fees -- so gating costs nothing in the P&L. Turnover is
   still reported, because a gate that doubles it would matter the moment that
   assumption is relaxed.

ONE APPROXIMATION, STATED
--------------------------
The gate is applied as a per-bar multiplier on realised book returns. For a
linear book at zero fees that is EXACT for the sizing itself. It does not
re-estimate the hedge overlay inside the gated path -- the hedge beta is fitted
on the ungated return history. That is a second-order effect on the level of
the hedge, not on the sign of the result, and it is flagged rather than hidden.

Both holdouts are spent. Every column here is a diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits, strategy
from mft.paths import DATA_DIR
from scripts.ridge import CAP
from scripts.run_strategy import attach_funding, cache_path, load_panel
from scripts.stage7_importance import LGB_PARAMS, SEEDS

WINDOWS = [3, 9, 30]                     # trailing bars for the forecast
KEEP = [1.00, 0.90, 0.80, 0.67, 0.50]    # fraction of bars left ON
WARMUP = 200                             # bars before the gate may act
BY = strategy.BARS_PER_YEAR


def trailing_pct(disp: pd.Series, win: int) -> pd.Series:
    """Expanding percentile of trailing dispersion. Strictly past-only:
    the rolling mean is shifted one bar, then ranked against its own history."""
    tr = disp.rolling(win, min_periods=max(2, win // 2)).mean().shift(1)
    return tr.expanding(min_periods=WARMUP).apply(
        lambda a: (a[:-1] < a[-1]).mean(), raw=True)


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    eq = (1 + r).cumprod()
    return {"sharpe": mu / sd * np.sqrt(BY) if sd else np.nan,
            "ann": float((1 + mu) ** BY - 1),
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "on": float((r != 0).mean())}


def book_returns(P: pd.DataFrame, seed=None, init_w=None):
    res = strategy.run(P, seed=seed, init_weights=init_w)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--out", default="features/dispersion_gate.json")
    a = ap.parse_args()
    import lightgbm as lgb

    lo = splits.TRUE_HOLDOUT2_MS
    Xtr, Ttr, cols, _ = load_panel(lo, lag=a.lag)
    Xte, Tte, _, _ = load_panel(CAP, lower_ms=lo, lag=a.lag)

    # ---- scores: OOF on train (cached), one forward fit for 2026 ---------
    CACHE = cache_path(a.lag)
    Ptr = pd.read_parquet(CACHE)
    print(f"DISPERSION GATE -- LightGBM scores, lag {a.lag}")
    print(f"  train book from cached OOF ({len(Ptr):,} rows)")
    res_tr = book_returns(Ptr)

    Rtr = Xtr.groupby(level="t_obs").rank(pct=True)
    Rte = Xte.groupby(level="t_obs").rank(pct=True)
    Ytr = Ttr["target"].groupby(level="t_obs").rank(pct=True).to_numpy()
    acc = np.zeros(len(Xte))
    for sd in SEEDS:
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
            Rtr.to_numpy("float32"), Ytr)
        q = m.predict(Rte.to_numpy("float32"))
        acc += (q - q.mean()) / (q.std() or 1.0)
    Pte = attach_funding(Tte.assign(score=acc / len(SEEDS)))
    seed_hist = (res_tr.ret.to_numpy(),
                 Ptr.groupby(level="t_obs")["fwd_rm"].first()
                 .reindex(res_tr.ret.index).to_numpy())
    iw = res_tr.weights.iloc[-1].reindex(
        sorted(Pte.index.get_level_values("symbol").unique())).fillna(0.0)
    res_te = book_returns(Pte, seed=seed_hist, init_w=iw.to_numpy())
    print(f"  2026 book {len(res_te.ret):,} bars\n")

    # ---- dispersion of REALISED residuals, per bar -----------------------
    # Realised at bar t is known only after t; the gate uses a TRAILING mean of
    # it, shifted, so nothing dated at or after t_obs enters the decision.
    def disp_of(T):
        return ((T["fwd_ret"] - T["beta"] * T["fwd_rm"])
                .groupby(level="t_obs").std().sort_index())

    # The 2026 gate must rank against TRAIN history, not restart its own
    # percentile scale -- otherwise the first 200 bars of 2026 are ungated and
    # the scale is recalibrated to the test period.
    d_all = pd.concat([disp_of(Ttr), disp_of(Tte)]).sort_index()

    out, base = {}, {}
    for lbl, r in (("train", res_tr.ret), ("2026", res_te.ret)):
        base[lbl] = stats(r)
    print(f"  {'gate':<22}" + "".join(f"{p:>24}" for p in ("train", "2026")))
    print(f"  {'':<22}" + "".join(f"{'Sharpe    ann    on%':>24}" for _ in range(2)))
    print("  " + "-" * 70)
    print(f"  {'UNGATED (baseline)':<22}"
          + "".join(f"{base[k]['sharpe']:>+10.2f}{base[k]['ann']:>9.1%}"
                    f"{100:>5.0f}" for k in ("train", "2026")))
    print()

    for win in WINDOWS:
        pct_all = trailing_pct(d_all, win)
        for keep in KEEP:
            if keep == 1.0:
                continue
            row = {}
            line = f"  win{win:<3} keep top {keep:<6.0%}"
            for lbl, r in (("train", res_tr.ret), ("2026", res_te.ret)):
                g = (pct_all.reindex(r.index) <= keep).astype(float)
                g = g.fillna(1.0)          # pre-warm-up: ungated, not dropped
                s = stats(r * g)
                row[lbl] = s
                line += f"{s['sharpe']:>+10.2f}{s['ann']:>9.1%}{s['on']*100:>5.0f}"
            print(line)
            out[f"win{win}_keep{keep:g}"] = row
        print()

    # ---- continuous inverse-dispersion sizing, no threshold at all -------
    print("  CONTINUOUS SIZING (no threshold picked): gross ~ 1 - pct")
    for win in WINDOWS:
        pct_all = trailing_pct(d_all, win)
        line = f"  win{win:<3} linear taper   "
        row = {}
        for lbl, r in (("train", res_tr.ret), ("2026", res_te.ret)):
            g = (1.0 - pct_all.reindex(r.index)).fillna(0.5).clip(0, 1) * 2
            s = stats(r * g)
            row[lbl] = s
            line += f"{s['sharpe']:>+10.2f}{s['ann']:>9.1%}{'--':>5}"
        print(line)
        out[f"win{win}_taper"] = row

    print("\n" + "=" * 72)
    print("  Read the GRID, not the best cell. The gate is worth something")
    print("  only if it improves 2026 across neighbouring windows AND")
    print("  thresholds, and does not destroy train.")
    p = DATA_DIR / a.out
    p.write_text(json.dumps({"baseline": base, "gated": out}, indent=2,
                            default=float), encoding="utf-8")
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
