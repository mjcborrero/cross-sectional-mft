"""Execute the frozen strategy on a split. Introduces no decisions.

    python scripts/run_strategy.py --split train
    python scripts/run_strategy.py --split holdout1     (burns holdout 1)

TRAIN uses the out-of-fold scores already produced by the walk-forward folds,
so every bar is priced by a model that never saw it.

A HOLDOUT RUN IS IRREVERSIBLE. It fits one model on ALL of train and applies it
forward, carrying the strategy's own state across the boundary: the last held
weights and the hedge history. Restarting either from zero would discard
information the strategy genuinely had, and would understate turnover costs by
pretending the book was flat on day one.

The script refuses to run a holdout unless `--i-am-burning-a-holdout` is passed,
because a holdout can be spent exactly once and an accidental invocation is not
recoverable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import folds as foldmod
from mft import splits, strategy
from mft.featbuild import MS_HOUR, MS_MIN, GridContext
from mft.paths import DATA_DIR
from scripts.stage7_importance import LGB_PARAMS, SEED, SEEDS, FAMILIES
from scripts.backtest_train import funding_per_window

# Execution lag in minutes. 60 was the ORIGINAL default, sized from the old
# batch pipeline's cycle ledger (~50-65 min: backfill -> live-update -> bars ->
# live-panel -> targets). That ledger measured a SEQUENTIAL pipeline; with
# streaming ingestion and incremental features almost all of it disappears,
# which is why the lag is now a first-class CLI parameter rather than a
# constant. What does NOT disappear with parallel infrastructure:
#   * kline completeness -- [t, t+1min) does not exist until after t+1min
#   * build_forward_returns.asof() takes the last close STRICTLY BEFORE
#     t_fill, so below lag=2 the fill price is dated at or before the decision
#     instant. That is not conservative-vs-aggressive, it is incoherent.
#   * t_obs sits exactly on the funding settlement (00/08/16), the worst
#     instant in the window to cross the spread.
# Hence a floor of 2 and a defensible default of 5, not 0.
LAG_MIN_FLOOR = 2
DEFAULT_LAG = 60


def cache_path(lag: int) -> Path:
    """OOF cache is keyed BY LAG. Without this, a run at lag 5 silently reuses
    scores fitted against the lag-60 target -- same filename, different label,
    no error raised. Exactly the class of bug this project keeps finding."""
    return DATA_DIR / (f"features/oof_book_inputs_lag{lag}.parquet" if lag != 60
                       else "features/oof_book_inputs.parquet")


def load_panel(upper_ms: int | None, lower_ms: int = 0,
               lag: int = DEFAULT_LAG) -> pd.DataFrame:
    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])
    parts = [pd.read_parquet(DATA_DIR / f"features/{f}.parquet")
             .set_index(["t_obs", "symbol"]) for f in FAMILIES.values()
             if (DATA_DIR / f"features/{f}.parquet").exists()]
    X = pd.concat(parts, axis=1)[cols]
    tgt = DATA_DIR / f"grid/target_8h_lag{lag}.parquet"
    if not tgt.exists():
        raise SystemExit(f"missing {tgt.name} -- build it first:\n"
                         f"  python scripts/build_forward_returns.py "
                         f"--lags {lag} --primary {lag}\n"
                         f"  python scripts/build_target.py --forward "
                         f"grid/forward_8h_lag{lag}.parquet "
                         f"--out grid/target_8h_lag{lag}.parquet")
    T = pd.read_parquet(tgt).set_index(["t_obs", "symbol"])
    i = X.index.intersection(T.index)
    X, T = X.loc[i].sort_index(), T.loc[i].sort_index()
    t = X.index.get_level_values("t_obs")
    m = (t >= lower_ms) & ((t < upper_ms) if upper_ms else True)
    return X[m], T[m], cols, fz


def attach_funding(P: pd.DataFrame) -> pd.DataFrame:
    tob = np.sort(P.index.get_level_values("t_obs").unique())
    f = funding_per_window(
        tob, sorted(P.index.get_level_values("symbol").unique())).stack()
    f.index = f.index.set_names(["t_obs", "symbol"])
    P = P.join(f.rename("funding"), how="left")
    P["funding"] = P["funding"].fillna(0.0)
    return P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    choices=["train", "holdout1", "holdout2"])
    ap.add_argument("--i-am-burning-a-holdout", action="store_true")
    ap.add_argument("--out", default=None)
    # An explicit right-hand cap on the holdout window, YYYY-MM-DD. Used for
    # holdout 2 because Binance's funding ARCHIVE lags: prices, spot, premium
    # and metrics reach 2026-08-29 but funding stops between 2026-07-31 and
    # 2026-08-04 by symbol. Funding feeds 7 of the 72 frozen features AND the
    # P&L, so August would be evaluated on stale carry. The cap is a DATA
    # COMPLETENESS decision, fixed before any 2026 result was seen, and it is
    # recorded in the output.
    ap.add_argument("--end", default=None,
                    help="cap the holdout window, YYYY-MM-DD (exclusive)")
    # Move the train/test boundary. DIAGNOSTIC ONLY: both holdouts are spent,
    # so a boundary chosen after seeing which month was bad cannot produce
    # out-of-sample evidence. It can answer whether the model needed more
    # recent data, which is a different and still-useful question.
    ap.add_argument("--train-end", default=None,
                    help="override the train/test boundary, YYYY-MM-DD "
                         "(DIAGNOSTIC: not out-of-sample)")
    # Run a recorded VARIANT instead of the active spec. Both variants were
    # selected on TRAIN evidence in the fee-ceiling sweep and written into
    # strategy.VARIANTS before either holdout was read, so running one is not
    # selection on holdout data.
    ap.add_argument("--variant", default=None,
                    choices=sorted(strategy.VARIANTS))
    ap.add_argument("--lag", type=int, default=DEFAULT_LAG,
                    help="execution lag in minutes (default 60). Lower values "
                         "recover short-horizon reversal that the 60min lag "
                         "excludes; see the header note before trusting them.")
    args = ap.parse_args()
    if args.lag < LAG_MIN_FLOOR:
        print(f"REFUSING lag={args.lag}. Floor is {LAG_MIN_FLOOR} min: "
              f"asof() fills at the last close STRICTLY BEFORE t_fill, so "
              f"below {LAG_MIN_FLOOR} the entry price is dated at or before "
              f"the decision instant that caused the trade.")
        return 2
    CACHE = cache_path(args.lag)
    if args.variant:
        v = strategy.VARIANTS[args.variant]
        strategy.LAMBDA, strategy.BAND = v["lam"], v["band"]
        print(f"  VARIANT '{args.variant}': lam {v['lam']}, band {v['band']}")
        print(f"    recorded -- train Sharpe {v['train']['sharpe']:+.2f}, "
              f"2025 Sharpe {v['h2025']['sharpe']:+.2f}")
    end_cap = (int(pd.Timestamp(args.end, tz="UTC").timestamp() * 1000)
               if args.end else None)

    if args.split != "train" and not args.i_am_burning_a_holdout:
        print(f"REFUSING to run {args.split}. A holdout can be spent once.")
        print("Pass --i-am-burning-a-holdout if that is genuinely the intent.")
        return 2

    spec = {k: getattr(strategy, k) for k in
            ("LAMBDA", "BAND", "GROSS", "HEDGE", "HEDGE_MIN_BARS", "MIN_COINS")}
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    print("FROZEN STRATEGY -- book C, dollar-neutral + hedge overlay")
    print(f"  lam {strategy.LAMBDA}, band {strategy.BAND}, gross "
          f"{strategy.GROSS}, spec hash {h}")
    print(f"  split: {args.split}\n")

    if args.split == "train":
        import lightgbm as lgb
        if CACHE.exists():
            P = pd.read_parquet(CACHE)
            print(f"  reusing cached out-of-fold scores ({len(P):,} rows)")
        else:
            X, T, cols, fz = load_panel(splits.HOLDOUT1_START_MS, lag=args.lag)
            Rk = X.groupby(level="t_obs").rank(pct=True)
            Yr = T["target"].groupby(level="t_obs").rank(pct=True)
            bars = X.index.get_level_values("t_obs").to_numpy()
            # SEED ENSEMBLE: average the predictions of every declared seed.
            # The book consumes only within-bar ORDER, and averaging the
            # scores averages the orderings, which is the point -- no single
            # arbitrary draw decides the book.
            sc = pd.Series(0.0, index=X.index)
            cnt = pd.Series(0.0, index=X.index)
            for f in foldmod.make_folds(np.sort(np.unique(bars))):
                tr, te = np.isin(bars, f.train), np.isin(bars, f.test)
                acc = np.zeros(int(te.sum()))
                for sd in SEEDS:
                    m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
                        Rk[tr].to_numpy("float32"), Yr[tr].to_numpy())
                    p_ = m.predict(Rk[te].to_numpy("float32"))
                    acc += (p_ - p_.mean()) / (p_.std() or 1.0)
                sc[te] = acc / len(SEEDS)
                cnt[te] = 1.0
            sc[cnt == 0] = np.nan
            P = attach_funding(T.assign(score=sc).dropna(subset=["score"]))
            # Persist: the out-of-fold panel is expensive (8 seeds x 5 folds)
            # and several downstream analyses need the identical scores, not
            # merely equivalent ones.
            P[["score", "fwd_ret", "fwd_rm", "beta", "sigma_eps",
               "funding"]].to_parquet(CACHE)
            print(f"  wrote {CACHE.name}")
        res = strategy.run(P)
    else:
        import lightgbm as lgb
        # TRUE_* deliberately, NOT HOLDOUT1_START_MS. The wall override moves
        # the latter to 2026, which makes the holdout empty -- the split
        # boundaries are calendar facts and must survive opening the wall.
        lo = (splits.TRUE_HOLDOUT1_MS if args.split == "holdout1"
              else splits.TRUE_HOLDOUT2_MS)
        hi = splits.TRUE_HOLDOUT2_MS if args.split == "holdout1" else None
        if args.train_end:
            lo = int(pd.Timestamp(args.train_end, tz="UTC").timestamp() * 1000)
            hi = None
            print(f"  *** DIAGNOSTIC: train/test boundary moved to "
                  f"{splits._fmt(lo)}. Not out-of-sample -- the boundary was "
                  f"chosen after seeing the outcome. ***")
        if end_cap is not None:
            hi = end_cap if hi is None else min(hi, end_cap)
            print(f"  evaluation capped at {splits._fmt(end_cap)} "
                  f"(data completeness, decided before any result)")
        if lo >= splits.HOLDOUT2_START_MS and args.split == "holdout1":
            raise SystemExit("holdout boundary resolved past 2026 -- refusing")
        # TRAIN is everything before the split being evaluated. One fit on all
        # of it, applied forward. No refitting inside the holdout -- that would
        # be a different strategy from the one that was frozen.
        Xtr, Ttr, cols, fz = load_panel(lo, lag=args.lag)
        Xho, Tho, _, _ = load_panel(hi, lower_ms=lo, lag=args.lag)
        print(f"  train {len(Xtr):,} rows to {splits._fmt(lo)}; "
              f"holdout {len(Xho):,} rows, "
              f"{Xho.index.get_level_values('t_obs').nunique():,} bars")

        Rtr = Xtr.groupby(level="t_obs").rank(pct=True)
        Rho = Xho.groupby(level="t_obs").rank(pct=True)
        Ytr = Ttr["target"].groupby(level="t_obs").rank(pct=True)
        acc = np.zeros(len(Xho))
        for sd in SEEDS:
            mdl = lgb.LGBMRegressor(**LGB_PARAMS, random_state=sd).fit(
                Rtr.to_numpy("float32"), Ytr.to_numpy())
            p_ = mdl.predict(Rho.to_numpy("float32"))
            acc += (p_ - p_.mean()) / (p_.std() or 1.0)
        P = attach_funding(Tho.assign(score=acc / len(SEEDS)))

        # Carry the strategy's own state across the boundary. Restarting the
        # hedge regression from zero would throw away four years of it, and
        # restarting the book from cash would understate the first bar's
        # turnover by pretending nothing was held.
        seed_hist, init_w = None, None
        if CACHE.exists():
            Ptr = pd.read_parquet(CACHE)
            rtr = strategy.run(Ptr)
            seed_hist = (rtr.ret.to_numpy(),
                         Ptr.groupby(level="t_obs")["fwd_rm"].first()
                         .reindex(rtr.ret.index).to_numpy())
            last = rtr.weights.iloc[-1]
            init_w = last.reindex(
                sorted(P.index.get_level_values("symbol").unique())
            ).fillna(0.0).to_numpy()
            print(f"  carried state: {len(seed_hist[0]):,} bars of hedge "
                  f"history, book gross {np.abs(init_w).sum():.3f}")
        res = strategy.run(P, seed=seed_hist, init_weights=init_w)

    # ---- monthly and annual breakdown --------------------------------
    idx = pd.to_datetime(res.ret.index, unit="ms")
    per = pd.DataFrame({"r": res.ret.to_numpy(), "t": res.turnover.to_numpy()},
                       index=idx)
    print("MONTHLY")
    print("=" * 84)
    print(f"  {'month':<9} {'bars':>5} {'return':>9} {'Sharpe':>8} {'vol':>8} "
          f"{'maxDD':>8} {'hit':>7} {'turn':>7} {'@2bp':>8}")
    rows_m = []
    for mo, gdf in per.groupby(per.index.to_period("M")):
        mm = strategy.metrics(gdf["r"], turnover=float(gdf["t"].mean()))
        cum = float((1 + gdf["r"]).prod() - 1)
        rows_m.append({"month": str(mo), "bars": mm["bars"], "return": cum,
                       "sharpe": mm["sharpe"], "ann_vol": mm["ann_vol"],
                       "max_drawdown": mm["max_drawdown"],
                       "hit_rate": mm["hit_rate"], "turnover": mm["turnover"],
                       })
        print(f"  {str(mo):<9} {mm['bars']:>5} {cum:>+9.2%} {mm['sharpe']:>+8.2f} "
              f"{mm['ann_vol']:>8.1%} {mm['max_drawdown']:>+8.2%} "
              f"{mm['hit_rate']:>7.1%} {mm['turnover']:>7.1%}")
    pos = sum(1 for r in rows_m if r["return"] > 0)
    print(f"\n  {pos}/{len(rows_m)} months positive")

    print("\nANNUAL")
    print("=" * 84)
    rows_y = []
    for yr, gdf in per.groupby(per.index.year):
        ym = strategy.metrics(gdf["r"], turnover=float(gdf["t"].mean()))
        cum = float((1 + gdf["r"]).prod() - 1)
        rows_y.append({"year": int(yr), **{k: ym[k] for k in
                       ("bars", "sharpe", "ann_vol", "max_drawdown",
                        "hit_rate", "turnover", "breakeven_bps")},
                       "return": cum})
        print(f"  {yr}  bars {ym['bars']:>5}  return {cum:>+8.2%}  "
              f"Sharpe {ym['sharpe']:>+6.2f}  vol {ym['ann_vol']:>6.1%}  "
              f"maxDD {ym['max_drawdown']:>+7.2%}")
        print(f"        hit {ym['hit_rate']:.1%}  turnover {ym['turnover']:.1%}  "
              f"break-even {ym['breakeven_bps']:.2f} bps")

    print()
    m = strategy.metrics(res.ret, turnover=float(res.turnover.mean()))
    print("OVERALL")
    print("=" * 74)
    print(f"  ann return {m['ann_return']:+.2%}   vol {m['ann_vol']:.2%}   "
          f"Sharpe {m['sharpe']:+.2f}")
    print(f"  hit {m['hit_rate']:.1%}   maxDD {m['max_drawdown']:+.2%}   "
          f"{m['bars']:,} bars")
    print(f"  turnover {m['turnover']:.1%} per rebalance   "
          f"break-even {m['breakeven_bps']:.2f} bps/side")
    # No fee grid. Fees are ZERO by design assumption, so the Sharpe above is
    # THE Sharpe. Turnover and break-even are printed as measurements of the
    # book -- they are what would let the assumption be revisited -- and not as
    # caveats attached to the headline.

    rm = P.groupby(level="t_obs")["fwd_rm"].first().reindex(res.ret.index)
    d = pd.DataFrame({"r": res.ret, "rm": rm, "a": rm.abs()}).dropna()
    for lbl, col in [("r_m  ", "rm"), ("|r_m|", "a")]:
        cc = float(np.corrcoef(d["r"], d[col])[0, 1])
        tt = cc * np.sqrt(len(d) - 2) / np.sqrt(max(1e-12, 1 - cc * cc))
        print(f"  corr(book, {lbl}) {cc:+.4f}  t {tt:+.2f}")
    print(f"  mean hedge beta {res.hedge_beta.mean():+.4f}")

    out = DATA_DIR / (args.out or f"features/strategy_{args.split}.json")
    out.write_text(json.dumps({"split": args.split, "spec": spec,
                               "end_cap": args.end,
                               "spec_hash": h, "metrics": m,
                               "monthly": rows_m, "annual": rows_y}, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
