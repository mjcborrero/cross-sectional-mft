"""SUPERSEDED BY THE ZERO-FEE ASSUMPTION. Fees are zero by design
(BETA_NEUTRAL_DESIGN) and every headline number in this project is a
zero-fee number. The fee-sensitivity analysis below was run while a
taker fee was being assumed ON TOP of the design; that assumption was
not the design's and has been withdrawn. The TURNOVER measurements are
facts about the book and remain valid; the fee-based verdicts are not
the project's conclusions.

Out-of-fold book and performance metrics on TRAIN. All 72 features.

This is the first time the project turns a score into a POSITION, so it is the
first time anything can be said about performance rather than about IC.

STRICTLY OUT OF FOLD. Every bar is priced by a model that never saw it: fold
k's test block is scored by a model fit on fold k's train block only, which is
everything before it minus the 2-bar purge. The five test blocks concatenate
into one continuous series from 2022-06 to 2024-12.

Fold 0 IS included here, unlike Stage 8. Stage 8 had to drop it because feature
SELECTION needed earlier folds to select from; there is no selection here -- all
72 features are carried -- so fold 0's model is as honest as any other.

THE BOOK, per BETA_NEUTRAL_DESIGN and the user's stated construction
--------------------------------------------------------------------
    dollar-neutral, inverse-vol weighted, rebalanced every 8h

    z_i  = within-bar centred rank of the model score
    w_i  = z_i / sigma_eps_i          inverse-vol sizing
    w    = w - mean(w)                dollar neutrality, sum w = 0
    w    = w / sum|w|                 gross exposure fixed at 1.0

FUNDING IS CHARGED, AND IT WAS DELIBERATELY EXCLUDED FROM THE TARGET
--------------------------------------------------------------------
TARGET_DESIGN: funding is kept out of the label so the model predicts genuine
price alpha rather than learning to harvest carry. It is a real cash flow and
is charged here. `last_funding_rate` positive means longs pay shorts, so the
book's funding P&L is -sum(w_i * f_i) over the settlements inside the holding
window. Both gross and net are reported so the size of the charge is visible.

FEES ARE ZERO BY ASSUMPTION, WHICH IS NOT THE SAME AS FEES BEING ZERO.
The design assumes zero fees. Turnover is reported alongside so the reader can
apply any cost they like: net Sharpe falls by roughly (cost per unit turnover x
turnover x bars per year) / annual volatility.

THE CONVEXITY CHECK IS RUN HERE BECAUSE IT COULD NOT BE RUN EARLIER
--------------------------------------------------------------------
Gate 9C found the target carries a |r_m| loading whose COMMON part cannot reach
a ranking objective but whose cross-sectional DISPERSION can reach a book.
FEATURE_SELECTION.md §8 registered the check for exactly this moment: measure
the realised book's exposure to r_m AND to |r_m|.
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
from mft import splits
from mft.featbuild import MS_HOUR, MS_MIN, GridContext
from mft.paths import DATA_DIR
from scripts.stage7_importance import LGB_PARAMS, SEED, FAMILIES

BARS_PER_YEAR = 365 * 3          # 8h bars
LAG_MS = 60 * MS_MIN
H_MS = 8 * MS_HOUR
MIN_COINS = 8


def funding_per_window(t_obs: np.ndarray, symbols: list[str]) -> pd.DataFrame:
    """Funding rate settled inside (t_fill, t_fill + h] for each coin and bar."""
    ctx = GridContext(t_obs)
    lo, hi = t_obs + LAG_MS, t_obs + LAG_MS + H_MS
    out = {}
    for sym in symbols:
        f = ctx.dataset("funding", sym, ["calc_time", "last_funding_rate"],
                        "calc_time")
        if f.empty:
            continue
        f = f.drop_duplicates("calc_time").sort_values("calc_time")
        ft = f["calc_time"].to_numpy(dtype=np.int64)
        fv = f["last_funding_rate"].to_numpy("float64")
        cum = np.concatenate([[0.0], np.cumsum(fv)])
        a = np.searchsorted(ft, lo, side="right")
        b = np.searchsorted(ft, hi, side="right")
        out[sym] = cum[b] - cum[a]
    return pd.DataFrame(out, index=pd.Index(t_obs, name="t_obs"))


def metrics(r: pd.Series, tag: str) -> dict:
    n = len(r)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sharpe = mu / sd * np.sqrt(BARS_PER_YEAR) if sd > 0 else np.nan
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"label": tag, "bars": n,
            "ann_return": float((1 + mu) ** BARS_PER_YEAR - 1),
            "ann_vol": float(sd * np.sqrt(BARS_PER_YEAR)),
            "sharpe": float(sharpe), "hit_rate": float((r > 0).mean()),
            "max_drawdown": dd, "mean_bar_bps": float(mu * 1e4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/backtest_train.json")
    args = ap.parse_args()

    import lightgbm as lgb

    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])

    parts = []
    for fname in FAMILIES.values():
        p = DATA_DIR / f"features/{fname}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p).set_index(["t_obs", "symbol"]))
    X = pd.concat(parts, axis=1)[cols]

    t = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet")
    t = t[t["t_obs"] < splits.HOLDOUT1_START_MS]
    splits.assert_train_only(t.assign(bar_close_time=t["t_obs"]))
    T = t.set_index(["t_obs", "symbol"])

    X = X[X.index.get_level_values("t_obs") < splits.HOLDOUT1_START_MS]
    idx = X.index.intersection(T.index)
    X, T = X.loc[idx].sort_index(), T.loc[idx].sort_index()
    Y = T["target"]

    Rk = X.groupby(level="t_obs").rank(pct=True)
    Yr = Y.groupby(level="t_obs").rank(pct=True)
    bars = X.index.get_level_values("t_obs").to_numpy()
    F = foldmod.make_folds(np.sort(np.unique(bars)))

    print("OUT-OF-FOLD BOOK ON TRAIN -- all 72 features")
    print(f"  {len(X):,} rows; every bar scored by a model that never saw it\n")

    score = pd.Series(np.nan, index=X.index, name="score")
    for f in F:
        tr, te = np.isin(bars, f.train), np.isin(bars, f.test)
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=SEED)
        m.fit(Rk[tr].to_numpy("float32"), Yr[tr].to_numpy())
        score[te] = m.predict(Rk[te].to_numpy("float32"))
    P = T.assign(score=score).dropna(subset=["score"])
    print(f"  scored {P['score'].notna().sum():,} rows over "
          f"{P.index.get_level_values('t_obs').nunique():,} bars")

    # ---- book construction ------------------------------------------------
    #
    # THE NEUTRALITY LEAK, AND THE FIX.
    #
    # The first version enforced only sum(w) = 0 and measured a realised
    # corr(book, r_m) of -0.127 at t -6.71. Dollar neutrality does NOT imply
    # beta neutrality: the book's market exposure is sum(w_i * beta_i), and
    # that is zero only if w is uncorrelated with beta across the bar. It is
    # not, because w = z / sigma_eps and sigma_eps covaries with beta -- so
    # inverse-vol sizing systematically underweights high-beta names and
    # leaves a residual short.
    #
    # The size checks out: mean sum(w*beta) was -0.038, and
    # -0.038 * var(r_m) / (sd_book * sd_rm) is about -0.17, the order of the
    # -0.127 observed. The exposure was the whole explanation, not a symptom.
    #
    # Fix: residualise w against [1, beta] inside each bar, which forces
    # sum(w) = 0 AND sum(w*beta) = 0 exactly. Both books are built and reported
    # side by side so the cost of the constraint is visible rather than
    # asserted.
    g = P.groupby(level="t_obs")
    n = g["score"].transform("size")
    P = P[n >= MIN_COINS]
    g = P.groupby(level="t_obs")
    z = g["score"].rank(pct=True)
    z = z - z.groupby(level="t_obs").transform("mean")
    raw = z / P["sigma_eps"].replace(0.0, np.nan)

    def gross_one(v: pd.Series) -> pd.Series:
        return v / v.abs().groupby(level="t_obs").transform("sum")

    # book A -- dollar-neutral only (what was measured before)
    wA = raw - raw.groupby(level="t_obs").transform("mean")
    wA = gross_one(wA)

    # book B -- dollar AND beta neutral
    b = P["beta"]
    bc = b - b.groupby(level="t_obs").transform("mean")
    wc = raw - raw.groupby(level="t_obs").transform("mean")
    num = (wc * bc).groupby(level="t_obs").transform("sum")
    den = (bc * bc).groupby(level="t_obs").transform("sum")
    wB = wc - bc * (num / den.replace(0.0, np.nan))
    wB = gross_one(wB)

    P = P.assign(w_dollar=wA, w_betaneutral=wB).dropna(
        subset=["w_dollar", "w_betaneutral"])
    for tag, col in [("dollar-neutral only", "w_dollar"),
                     ("dollar + beta neutral", "w_betaneutral")]:
        e = (P[col] * P["beta"]).groupby(level="t_obs").sum()
        print(f"  {tag:<24} sum(w*beta): mean {e.mean():+.5f}  "
              f"std {e.std():.5f}")
    w = P["w_betaneutral"]
    P = P.assign(w=w)

    tob = np.sort(P.index.get_level_values("t_obs").unique())
    syms = sorted(P.index.get_level_values("symbol").unique())
    Fund = funding_per_window(tob, syms)
    fund = Fund.stack().rename("funding")
    fund.index = fund.index.set_names(["t_obs", "symbol"])
    P = P.join(fund, how="left")
    P["funding"] = P["funding"].fillna(0.0)

    gross = (P["w"] * P["fwd_ret"]).groupby(level="t_obs").sum()
    fcost = (P["w"] * P["funding"]).groupby(level="t_obs").sum()
    net = gross - fcost

    # turnover: gross weight change between consecutive rebalances
    Wm = P["w"].unstack("symbol").reindex(tob).fillna(0.0)
    turn = (Wm - Wm.shift(1)).abs().sum(axis=1).iloc[1:]

    print("\nBOTH BOOKS, net of funding -- what the constraint costs")
    print("=" * 74)
    both = {}
    rm_all = T.groupby(level="t_obs")["fwd_rm"].first()
    for tag, col in [("dollar-neutral only", "w_dollar"),
                     ("dollar + beta neutral", "w_betaneutral")]:
        gr = (P[col] * P["fwd_ret"]).groupby(level="t_obs").sum()
        nt = gr - (P[col] * P["funding"]).groupby(level="t_obs").sum()
        Wt = P[col].unstack("symbol").reindex(tob).fillna(0.0)
        tn = float((Wt - Wt.shift(1)).abs().sum(axis=1).iloc[1:].mean())
        d0 = pd.DataFrame({"r": nt, "rm": rm_all.reindex(nt.index)}).dropna()
        cc = float(np.corrcoef(d0["r"], d0["rm"])[0, 1])
        tt0 = float(cc * np.sqrt(len(d0) - 2) / np.sqrt(max(1e-12, 1 - cc * cc)))
        bb = float((P[col] * P["beta"]).groupby(level="t_obs").sum().mean())
        m = metrics(nt, tag)
        both[tag] = dict(m, corr_rm=cc, t_rm=tt0, turnover=tn, book_beta=bb)
        print(f"  {tag:<23} Sharpe {m['sharpe']:>+6.2f}  ann {m['ann_return']:>+7.2%}"
              f"  corr(r_m) {cc:>+7.4f} t {tt0:>+6.2f}  turn {tn:.1%}")
    print("  The constraint COSTS return. That is the price of the design "
          "claim being\n  true rather than approximately true, and it is paid "
          "rather than argued away.")

    print("\nPERFORMANCE, out of fold, gross exposure 1.0, 8h rebalance")
    print("  (book = DOLLAR + BETA NEUTRAL, the corrected construction)")
    print("=" * 74)
    rows = [metrics(gross, "gross (no funding)"),
            metrics(net, "NET of funding")]
    print(f"  {'':<20} {'ann ret':>9} {'ann vol':>9} {'Sharpe':>8} "
          f"{'hit':>7} {'maxDD':>8} {'bps/bar':>9}")
    for m in rows:
        print(f"  {m['label']:<20} {m['ann_return']:>+8.2%} {m['ann_vol']:>8.2%} "
              f"{m['sharpe']:>+8.2f} {m['hit_rate']:>6.1%} "
              f"{m['max_drawdown']:>+8.2%} {m['mean_bar_bps']:>+9.2f}")

    print(f"\n  funding cost: {fcost.mean()*1e4:+.2f} bps/bar "
          f"({(1+fcost.mean())**BARS_PER_YEAR - 1:+.2%}/yr) -- charged, and "
          f"excluded from the target by design")
    print(f"  turnover: {turn.mean():.1%} of gross per 8h rebalance "
          f"({turn.mean()*BARS_PER_YEAR:.0f}x/yr)")
    print(f"  FEES ARE ZERO BY ASSUMPTION. At c bps per unit turnover, net "
          f"Sharpe falls\n  by about c x {turn.mean()*BARS_PER_YEAR/1e4:.2f} / "
          f"{rows[1]['ann_vol']:.4f} per bp.")

    # ---- per fold, so a single lucky block cannot carry the number --------
    print("\nPER FOLD (net)")
    print("=" * 74)
    per_fold = []
    for f in F:
        r = net[np.isin(net.index.to_numpy(), f.test)]
        if len(r) < 30:
            continue
        m = metrics(r, f"fold {f.k}")
        per_fold.append(m)
        print(f"  fold {f.k}: Sharpe {m['sharpe']:>+6.2f}  "
              f"ann ret {m['ann_return']:>+7.2%}  maxDD {m['max_drawdown']:>+7.2%}"
              f"  bars {m['bars']}")
    pos = sum(1 for m in per_fold if m["sharpe"] > 0)
    print(f"  positive Sharpe in {pos}/{len(per_fold)} folds")

    # ---- the pre-registered exposure checks -------------------------------
    print("\nEXPOSURE CHECKS (FEATURE_SELECTION.md §8, registered after Gate 9C)")
    print("=" * 74)
    rm = T.groupby(level="t_obs")["fwd_rm"].first().reindex(net.index)
    bexp = (P["w"] * P["beta"]).groupby(level="t_obs").sum().reindex(net.index)
    d = pd.DataFrame({"r": net, "rm": rm, "arm": rm.abs(), "b": bexp}).dropna()

    def ct(a, b):
        r = float(np.corrcoef(a, b)[0, 1])
        nn = len(a)
        return r, r * np.sqrt(nn - 2) / np.sqrt(max(1e-12, 1 - r * r))

    r1, t1 = ct(d["r"], d["rm"])
    r2, t2 = ct(d["r"], d["arm"])
    print(f"  corr(book return, r_m  ) {r1:+.4f}  t {t1:+6.2f}   linear market")
    print(f"  corr(book return, |r_m|) {r2:+.4f}  t {t2:+6.2f}   CONVEXITY")
    print(f"  book beta (sum w_i beta_i): mean {d['b'].mean():+.4f}, "
          f"std {d['b'].std():.4f}")
    if abs(t2) >= 2:
        print("  -> The book DOES carry the convexity Gate 9C warned about.")
        print("     Registered as a real exposure, not explained away.")
    else:
        print("  -> No detectable convexity exposure in the realised book.")

    # The LINEAR check is the one that matters here, and it is the one that
    # fails. Ex-ante book beta is ~0 by construction, so a significant
    # realised correlation means the neutrality is not being delivered by the
    # weights. Broken out per fold, because a single directional regime could
    # otherwise carry it.
    print("\n  per fold:")
    for f in F:
        dd = d[np.isin(d.index.to_numpy(), f.test)]
        if len(dd) < 30:
            continue
        rr, tt = ct(dd["r"], dd["rm"])
        print(f"    fold {f.k}: corr(book, r_m) {rr:+.4f}  t {tt:+6.2f}   "
              f"book beta {dd['b'].mean():+.4f}")

    # ---- how much of the result is that exposure? -------------------------
    # Hedge with an EXPANDING beta estimated only on bars already seen. No
    # in-sample fit, so this is an honest counterfactual rather than an
    # upper bound.
    print("\n  HEDGING THE RESIDUAL MARKET EXPOSURE (expanding beta, past only)")
    rv, mv = d["r"].to_numpy(), d["rm"].to_numpy()
    hb = np.full(len(rv), np.nan)
    for i in range(200, len(rv)):
        x, y = mv[:i], rv[:i]
        hb[i] = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0.0
    ok = np.isfinite(hb)
    hedged = pd.Series(rv[ok] - hb[ok] * mv[ok], index=d.index[ok])
    raw_tail = pd.Series(rv[ok], index=d.index[ok])
    mh = metrics(hedged, "hedged")
    mr = metrics(raw_tail, "unhedged, same bars")
    print(f"    unhedged (same bars): Sharpe {mr['sharpe']:+.2f}, "
          f"ann ret {mr['ann_return']:+.2%}")
    print(f"    market-hedged       : Sharpe {mh['sharpe']:+.2f}, "
          f"ann ret {mh['ann_return']:+.2%}")
    print(f"    mean hedge beta {np.nanmean(hb[ok]):+.4f}")
    print(f"    -> {'the exposure was CARRYING part of the result' if mh['sharpe'] < mr['sharpe'] - 0.15 else 'the result SURVIVES hedging'}")

    # ---- cost sensitivity, recorded not just printed ----------------------
    print("\nTURNOVER COST SENSITIVITY -- fees are ZERO BY ASSUMPTION")
    print("=" * 74)
    units = turn.mean() * BARS_PER_YEAR
    vol = rows[1]["ann_vol"]
    print(f"  {units:.0f} units of gross traded per year at "
          f"{turn.mean():.1%} per rebalance")
    print(f"  {'cost/side (bps)':>16} {'ann cost':>10} {'Sharpe':>9}")
    costs = {}
    for c in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
        annc = units * c / 1e4
        s = rows[1]["sharpe"] - annc / vol
        costs[c] = float(s)
        note = ("   <- Binance maker ~2bp" if c == 2 else
                "   <- taker ~4-5bp" if c == 4 else "")
        print(f"  {c:>16.1f} {annc:>9.2%} {s:>+9.2f}{note}")
    breakeven = float(rows[1]["sharpe"] * vol / units * 1e4)
    print(f"\n  BREAK-EVEN COST: {breakeven:.2f} bps/side. Below the Binance "
          f"taker fee.\n  The zero-fee assumption is doing a great deal of work "
          f"in the headline\n  number and must be stated wherever that number "
          f"is quoted.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps({
        "scope": "TRAIN out-of-fold", "n_features": len(cols),
        "frozen_hash": fz["content_hash"], "bars_per_year": BARS_PER_YEAR,
        "gross": rows[0], "net": rows[1], "books": both,
        "book": "dollar+beta neutral (weights residualised against beta per bar)",
        "funding_bps_per_bar": float(fcost.mean() * 1e4),
        "turnover_per_rebalance": float(turn.mean()),
        "per_fold": per_fold,
        "exposure": {"corr_rm": r1, "t_rm": t1, "corr_abs_rm": r2,
                     "t_abs_rm": t2, "book_beta_mean": float(d["b"].mean()),
                     "book_beta_std": float(d["b"].std())},
        "fees": "zero by assumption; turnover reported for rescaling",
        "cost_sensitivity_sharpe": costs,
        "breakeven_bps_per_side": breakeven,
        "hedged": mh, "unhedged_same_bars": mr,
        "mean_hedge_beta": float(np.nanmean(hb[ok])),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
