"""SUPERSEDED BY THE ZERO-FEE ASSUMPTION. Fees are zero by design
(BETA_NEUTRAL_DESIGN) and every headline number in this project is a
zero-fee number. The fee-sensitivity analysis below was run while a
taker fee was being assumed ON TOP of the design; that assumption was
not the design's and has been withdrawn. The TURNOVER measurements are
facts about the book and remain valid; the fee-based verdicts are not
the project's conclusions.

Book variants and the fee ceiling. TRAIN, out of fold.

TWO QUESTIONS, IN ORDER
-----------------------
1. Which book? Three constructions are compared, because the choice between the
   first two was being made on an incomplete comparison.
2. Given that book, how much turnover can be removed before the alpha goes?
   The strategy breaks even at ~3 bps/side, below the Binance taker fee, so the
   fee ceiling is the binding constraint on whether any of this is tradeable.

THE THREE BOOKS
---------------
    A  DOLLAR-NEUTRAL      sum(w) = 0 only. Highest Sharpe, but carries
                           corr(book, r_m) = -0.127 at t -6.71.
    B  BETA-CONSTRAINED    w residualised against [1, beta] per bar. Neutral by
                           construction; costs 0.49 of Sharpe because it moves
                           the positions and loses alpha with them.
    C  HEDGED OVERLAY      A's positions, unchanged, plus a market hedge sized
                           by an EXPANDING regression of book return on r_m
                           using past bars only. Neutralises the exposure
                           without touching the alpha.

C is the one the earlier run hinted at and did not name: hedging IMPROVED the
dollar-neutral book (+2.35 -> +2.61) while constraining it HURT (2.61 -> 2.12).
Those are different operations and conflating them was the error.

C's hedge leg trades, so its turnover is charged too. Nothing here is free.

TURNOVER REDUCTION
------------------
Two standard instruments, swept together:

    SMOOTHING   w_t = lam * w_{t-1} + (1 - lam) * w_target_t
                Trades a little signal freshness for a lot of turnover.

    NO-TRADE BAND
                Hold the previous weight unless the target differs by more than
                `band` of gross. Removes small pointless adjustments.

Reported as net Sharpe on a grid of realistic costs, so the choice is made
against the fee actually paid rather than against zero.
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
from scripts.backtest_train import (BARS_PER_YEAR, MIN_COINS, funding_per_window,
                                    metrics)

CACHE = DATA_DIR / "features/oof_book_inputs.parquet"
COSTS = [0.0, 1.0, 2.0, 4.0]


def build_inputs() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import lightgbm as lgb
    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])
    parts = [pd.read_parquet(DATA_DIR / f"features/{f}.parquet")
             .set_index(["t_obs", "symbol"]) for f in FAMILIES.values()
             if (DATA_DIR / f"features/{f}.parquet").exists()]
    X = pd.concat(parts, axis=1)[cols]
    t = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet")
    t = t[t["t_obs"] < splits.HOLDOUT1_START_MS]
    splits.assert_train_only(t.assign(bar_close_time=t["t_obs"]))
    T = t.set_index(["t_obs", "symbol"])
    X = X[X.index.get_level_values("t_obs") < splits.HOLDOUT1_START_MS]
    i = X.index.intersection(T.index)
    X, T = X.loc[i].sort_index(), T.loc[i].sort_index()
    Rk = X.groupby(level="t_obs").rank(pct=True)
    Yr = T["target"].groupby(level="t_obs").rank(pct=True)
    bars = X.index.get_level_values("t_obs").to_numpy()
    sc = pd.Series(np.nan, index=X.index)
    for f in foldmod.make_folds(np.sort(np.unique(bars))):
        tr, te = np.isin(bars, f.train), np.isin(bars, f.test)
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=SEED).fit(
            Rk[tr].to_numpy("float32"), Yr[tr].to_numpy())
        sc[te] = m.predict(Rk[te].to_numpy("float32"))
    P = T.assign(score=sc).dropna(subset=["score"])
    n = P.groupby(level="t_obs")["score"].transform("size")
    P = P[n >= MIN_COINS]
    tob = np.sort(P.index.get_level_values("t_obs").unique())
    fund = funding_per_window(
        tob, sorted(P.index.get_level_values("symbol").unique())).stack()
    fund.index = fund.index.set_names(["t_obs", "symbol"])
    P = P.join(fund.rename("funding"), how="left")
    P["funding"] = P["funding"].fillna(0.0)
    P = P[["score", "fwd_ret", "fwd_rm", "beta", "sigma_eps", "funding"]]
    P.to_parquet(CACHE)
    return P


def target_weights(P: pd.DataFrame, beta_neutral: bool) -> pd.Series:
    g = P.groupby(level="t_obs")
    z = g["score"].rank(pct=True)
    z = z - z.groupby(level="t_obs").transform("mean")
    w = (z / P["sigma_eps"].replace(0.0, np.nan))
    w = w - w.groupby(level="t_obs").transform("mean")
    if beta_neutral:
        bc = P["beta"] - P["beta"].groupby(level="t_obs").transform("mean")
        num = (w * bc).groupby(level="t_obs").transform("sum")
        den = (bc * bc).groupby(level="t_obs").transform("sum")
        w = w - bc * (num / den.replace(0.0, np.nan))
    return w / w.abs().groupby(level="t_obs").transform("sum")


def apply_smoothing(W: pd.DataFrame, lam: float, band: float) -> pd.DataFrame:
    """EWMA smoothing then a no-trade band, both on the weight matrix."""
    if lam > 0:
        W = W.ewm(alpha=1 - lam, adjust=False).mean()
        W = W.div(W.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
    if band <= 0:
        return W.fillna(0.0)
    A = W.to_numpy(copy=True)
    held = np.zeros(A.shape[1])
    for i in range(len(A)):
        tgt = np.nan_to_num(A[i])
        move = np.abs(tgt - held) > band
        held = np.where(move, tgt, held)
        A[i] = held
    return pd.DataFrame(A, index=W.index, columns=W.columns).fillna(0.0)


def run_book(P: pd.DataFrame, W: pd.DataFrame, hedge: bool) -> dict:
    """P&L, turnover and exposure for a weight matrix."""
    tob = W.index
    R = P["fwd_ret"].unstack("symbol").reindex(tob).reindex(columns=W.columns)
    Fn = P["funding"].unstack("symbol").reindex(tob).reindex(columns=W.columns)
    B = P["beta"].unstack("symbol").reindex(tob).reindex(columns=W.columns)
    rm = P.groupby(level="t_obs")["fwd_rm"].first().reindex(tob)

    gross = (W * R.fillna(0.0)).sum(axis=1)
    fund = (W * Fn.fillna(0.0)).sum(axis=1)
    net = gross - fund
    turn = (W - W.shift(1)).abs().sum(axis=1)
    turn.iloc[0] = W.abs().sum(axis=1).iloc[0]

    if hedge:
        # Expanding hedge ratio from PAST bars only. No in-sample fit.
        r, m = net.to_numpy(), rm.to_numpy()
        hb = np.zeros(len(r))
        for i in range(200, len(r)):
            v = np.var(m[:i])
            hb[i] = np.cov(m[:i], r[:i])[0, 1] / v if v > 0 else 0.0
        net = net - pd.Series(hb, index=tob) * rm
        # the overlay is a position too, and rebalancing it costs turnover
        turn = turn + pd.Series(np.abs(np.diff(hb, prepend=0.0)), index=tob)

    bexp = (W * B.fillna(0.0)).sum(axis=1)
    d = pd.DataFrame({"r": net, "rm": rm}).dropna()
    cc = float(np.corrcoef(d["r"], d["rm"])[0, 1])
    tt = float(cc * np.sqrt(len(d) - 2) / np.sqrt(max(1e-12, 1 - cc * cc)))
    m0 = metrics(net, "")
    tn = float(turn.mean())
    vol = m0["ann_vol"]
    return {**m0, "turnover": tn, "corr_rm": cc, "t_rm": tt,
            "book_beta": float(bexp.mean()),
            "net_sharpe": {c: float(m0["sharpe"] - (tn * BARS_PER_YEAR * c / 1e4) / vol)
                           for c in COSTS},
            "breakeven_bps": float(m0["sharpe"] * vol / (tn * BARS_PER_YEAR) * 1e4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/fee_ceiling.json")
    args = ap.parse_args()

    P = build_inputs()
    print("BOOK VARIANTS AND THE FEE CEILING -- TRAIN, out of fold")
    print(f"  {len(P):,} rows, "
          f"{P.index.get_level_values('t_obs').nunique():,} bars\n")

    wA = target_weights(P, beta_neutral=False).unstack("symbol").fillna(0.0)
    wB = target_weights(P, beta_neutral=True).unstack("symbol").fillna(0.0)

    print("THE THREE BOOKS, no smoothing, net of funding")
    print("=" * 78)
    print(f"  {'book':<26} {'Sharpe':>7} {'ann':>8} {'corr(r_m)':>10} {'t':>7} "
          f"{'turn':>6} {'b/e bps':>8}")
    books = {}
    for tag, W, hg in [("A dollar-neutral", wA, False),
                       ("B beta-constrained", wB, False),
                       ("C dollar + hedge overlay", wA, True)]:
        r = run_book(P, W, hg)
        books[tag] = r
        print(f"  {tag:<26} {r['sharpe']:>+7.2f} {r['ann_return']:>+8.2%} "
              f"{r['corr_rm']:>+10.4f} {r['t_rm']:>+7.2f} {r['turnover']:>5.0%} "
              f"{r['breakeven_bps']:>8.2f}")

    print("\n  Net Sharpe after fees:")
    print(f"  {'book':<26}" + "".join(f"{c:>9.0f}bp" for c in COSTS))
    for tag, r in books.items():
        print(f"  {tag:<26}" + "".join(f"{r['net_sharpe'][c]:>+11.2f}"
                                      for c in COSTS))

    # ---- turnover reduction on the chosen book --------------------------
    print("\n\nFEE CEILING -- turnover reduction on book A (dollar-neutral)")
    print("=" * 78)
    print(f"  {'lam':>5} {'band':>6} {'turn':>6} {'Sharpe':>8} "
          + "".join(f"{c:>8.0f}bp" for c in COSTS) + f"{'b/e':>8}")
    grid, best = [], None
    for lam in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]:
        for band in [0.0, 0.005, 0.02]:
            W = apply_smoothing(wA, lam, band)
            r = run_book(P, W, False)
            grid.append({"lam": lam, "band": band, **r})
            print(f"  {lam:>5.2f} {band:>6.3f} {r['turnover']:>5.0%} "
                  f"{r['sharpe']:>+8.2f} "
                  + "".join(f"{r['net_sharpe'][c]:>+10.2f}" for c in COSTS)
                  + f"{r['breakeven_bps']:>8.2f}")
            if best is None or r["net_sharpe"][4.0] > best["net_sharpe"][4.0]:
                best = {"lam": lam, "band": band, **r}

    # ---- the combination: hedged overlay AND smoothed ---------------------
    # The overlay leaves positions untouched, so it composes with smoothing.
    print("\n\nCOMBINED -- book C (dollar-neutral + hedge overlay) + smoothing")
    print("=" * 78)
    print(f"  {'lam':>5} {'band':>6} {'turn':>6} {'Sharpe':>8} "
          + "".join(f"{c:>8.0f}bp" for c in COSTS) + f"{'b/e':>8}")
    cgrid, cbest = [], None
    for lam in [0.0, 0.7, 0.8, 0.9, 0.95]:
        for band in [0.0, 0.005]:
            W = apply_smoothing(wA, lam, band)
            r = run_book(P, W, True)
            cgrid.append({"lam": lam, "band": band, **r})
            print(f"  {lam:>5.2f} {band:>6.3f} {r['turnover']:>5.0%} "
                  f"{r['sharpe']:>+8.2f} "
                  + "".join(f"{r['net_sharpe'][c]:>+10.2f}" for c in COSTS)
                  + f"{r['breakeven_bps']:>8.2f}")
            if cbest is None or r["net_sharpe"][4.0] > cbest["net_sharpe"][4.0]:
                cbest = {"lam": lam, "band": band, **r}

    print("\n" + "=" * 78)
    b = best
    print(f"BEST AT TAKER (4 bps): lam {b['lam']}, band {b['band']:.3f}")
    print(f"  turnover {b['turnover']:.1%} per rebalance "
          f"(was {books['A dollar-neutral']['turnover']:.1%})")
    print(f"  zero-fee Sharpe {b['sharpe']:+.2f}, "
          f"at 2bp {b['net_sharpe'][2.0]:+.2f}, at 4bp {b['net_sharpe'][4.0]:+.2f}")
    print(f"  break-even {b['breakeven_bps']:.2f} bps/side "
          f"(was {books['A dollar-neutral']['breakeven_bps']:.2f})")

    out = DATA_DIR / args.out
    print(f"\nBEST OVERALL AT TAKER: book C + lam {cbest['lam']}, "
          f"band {cbest['band']:.3f}")
    print(f"  turnover {cbest['turnover']:.1%}, zero-fee Sharpe "
          f"{cbest['sharpe']:+.2f}, at 2bp {cbest['net_sharpe'][2.0]:+.2f}, "
          f"at 4bp {cbest['net_sharpe'][4.0]:+.2f}")
    print(f"  corr(book, r_m) {cbest['corr_rm']:+.4f} (t {cbest['t_rm']:+.2f}), "
          f"break-even {cbest['breakeven_bps']:.2f} bps/side")
    out.write_text(json.dumps({"books": books, "grid": grid, "best_at_4bp": b,
                               "combined_grid": cgrid, "best_combined": cbest,
                               "costs_bps": COSTS}, indent=2, default=float),
                   encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
