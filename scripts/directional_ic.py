"""Does market-level predictability exist? A go/no-go gate, not a strategy.

WHY THIS IS THE FIRST THING TO MEASURE
---------------------------------------
IR ~ IC * sqrt(breadth). Directionally the universe is nearly ONE asset:
average pairwise coin correlation is 0.58 (train), 0.67 (2025), 0.62 (2026), so
20 positions are worth

    N / (1 + (N-1)*rho)  =  20 / (1 + 19*0.6)  ~  1.6 independent bets.

At breadth 1.6 the IC needed for IR 1.0 is

    8h      1095*1.6 = 1752 bets/yr   sqrt = 41.9   ->  IC 0.024
    daily    365*1.6 =  584           sqrt = 24.2   ->  IC 0.042
    weekly    52*1.6 =   83           sqrt =  9.1   ->  IC 0.110

Those are the bars. If nothing clears them, a directional book cannot reach
IR 1.0 no matter how it is engineered, and the answer is to stop.

CORRECTION (2026-09): THE BARS ARE TOO LENIENT, AND ARE NO LONGER THE CRITERION
-------------------------------------------------------------------------------
`IR ~ IC*sqrt(BR)` is an APPROXIMATION. For a time-series signal the P&L can be
computed directly -- hold position proportional to the z-scored signal and
measure the Sharpe -- so the approximation is unnecessary here. Doing both and
comparing the median ratio across every signal on train:

    horizon   bar implies IR per unit IC   measured Sharpe per unit IC   ratio
    8h                 41.9                        27.4                  1.53x
    24h                24.2                        16.3                  1.48x
    weekly              9.1                         7.4                  1.23x

The bar OVERSTATES achievable Sharpe by 1.2-1.5x, so a signal that "clears" it
does not in fact reach IR 1.0. The main cause is the breadth term: 1.6 is the
number of independent bets in 20 CROSS-SECTIONAL positions, but a market-wide
signal predicting the market return is ONE bet per period. sqrt(1.6) = 1.27
accounts for most of the gap; the remainder is autocorrelation and non-normality
that the formula ignores.

The `Sharpe` column is therefore the decision statistic. The bar and its `*`
flag are retained only because they were pre-registered, and where the two
disagree the direct measurement wins.

A NOTE ON WHAT WAS *NOT* WRONG HERE
------------------------------------
The cross-sectional path's `per_bar_ic` has a separate defect -- it ranks within
a bar (discarding magnitude) and then averages bars equally, so it reads
positive while the book loses: on the 8h book in 2026, plain IC +0.0274 vs
dispersion-weighted +0.0007. That defect does NOT apply to this script, which
uses Pearson on RAW values throughout, so large moves already carry
proportionate weight. The opposite risk applies instead, and gets the
`tail` column: a Pearson correlation on raw returns can be carried by a handful
of extreme bars. Above 35% the value is flagged `!`.

WHAT IS MEASURED
----------------
Signal at t against the FORWARD MARKET RETURN, not the idiosyncratic target.
This is a time-series correlation over bars, not a cross-sectional one.

Signals are market-wide aggregates built from artifacts that already passed
their gates -- trailing trend and vol from the index, and cross-sectional means
of audited features for carry, basis and premium. The three CONTEXT features
this project discarded (btc_resid_lag, eth_resid_lag, term_basis_slope) are
included on purpose: they were exempted from every cross-sectional stage for
having no ranking power, which is exactly what makes them directional
candidates.

OVERLAP IS AVOIDED BY SUBSAMPLING, NOT CORRECTED FOR
------------------------------------------------------
A k-bar forward return sampled every bar overlaps k-deep and inflates every
t-statistic. Each horizon is therefore evaluated on NON-OVERLAPPING bars only
(every 3rd for 24h, every 21st for weekly). That costs sample size and buys
honest t-statistics, which is the right trade when the whole point is a
go/no-go decision.

Train and 2026 are reported separately. Both holdouts are spent, so neither
column is out-of-sample; they are shown apart because a signal that only works
in one is not a signal.
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
from scripts.stage7_importance import FAMILIES

CAP = int(pd.Timestamp("2026-07-31", tz="UTC").timestamp() * 1000)
HORIZONS = {"8h": 1, "24h": 3, "weekly": 21}
BREADTH = 1.6
BARS_YR = 1095


def ic_t(x: np.ndarray, y: np.ndarray, per_year: float) -> dict:
    """Pearson IC plus the two things the IC alone does not tell you.

    NOTE ON WHAT THIS IS *NOT*. The cross-sectional path (`per_bar_ic`) has a
    known defect: it rank-transforms within a bar, discarding magnitude, and
    then averages bars equally -- so it can read positive while the book loses,
    because P&L weights bars by how much moved. Measured on the 8h book in
    2026: plain IC +0.0274, dispersion-weighted +0.0007.

    That defect does NOT apply here. This is Pearson on RAW values, so a large
    forward return already contributes proportionally more to the covariance.
    There is no equal-weighting of bars to undo.

    The failure mode here is the OPPOSITE one, and it gets its own column:
    Pearson on raw returns is outlier-driven, so a handful of extreme bars can
    carry the whole correlation. `tail_share` reports the fraction of the total
    |cross-product| contributed by the top 5% of bars by |y|. Above ~0.35 the
    correlation is a few days, not a signal.

    `sharpe` is the direct P&L statistic: the annualised Sharpe of holding
    position proportional to the z-scored signal. The `bar` these are compared
    against is a Grinold APPROXIMATION (IR ~ IC*sqrt(BR)); this is the quantity
    the approximation is standing in for, computed without it.
    """
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 40:
        return {"ic": np.nan, "t": np.nan, "n": n,
                "sharpe": np.nan, "tail_share": np.nan}
    xv, yv = x[m], y[m]
    c = float(np.corrcoef(xv, yv)[0, 1])
    t = float(c * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - c * c)))

    sd = xv.std(ddof=1)
    z = (xv - xv.mean()) / (sd if sd else 1.0)
    pnl = z * yv
    s = pnl.std(ddof=1)
    sharpe = float(pnl.mean() / s * np.sqrt(per_year)) if s else np.nan

    cp = np.abs((xv - xv.mean()) * (yv - yv.mean()))
    cut = np.quantile(np.abs(yv), 0.95)
    tot = cp.sum()
    tail = float(cp[np.abs(yv) >= cut].sum() / tot) if tot else np.nan
    return {"ic": c, "t": t, "n": n, "sharpe": sharpe, "tail_share": tail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/directional_ic.json")
    args = ap.parse_args()

    T = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet")
    rm = T.groupby("t_obs")["fwd_rm"].first().sort_index()
    t = np.asarray(rm.index)

    fz = json.loads((DATA_DIR / "features/frozen_list.json").read_text())
    cols = list(fz["features"])
    X = pd.concat([pd.read_parquet(DATA_DIR / f"features/{f}.parquet")
                   .set_index(["t_obs", "symbol"]) for f in FAMILIES.values()
                   if (DATA_DIR / f"features/{f}.parquet").exists()], axis=1)
    have = [c for c in cols if c in X.columns]
    M = X[have].groupby(level="t_obs").mean().reindex(rm.index)   # market-wide

    S = pd.DataFrame(index=rm.index)
    # --- trend and vol, from the index itself -------------------------
    lr = np.log1p(rm)
    for tag, w in [("trend_7d", 21), ("trend_30d", 90), ("trend_90d", 270)]:
        S[tag] = lr.rolling(w, min_periods=w // 2).sum().shift(1)
    S["vol_30d"] = rm.rolling(90, min_periods=45).std().shift(1)
    S["vol_ratio_7_30"] = (rm.rolling(21, min_periods=10).std()
                           / rm.rolling(90, min_periods=45).std()).shift(1)
    # --- market-wide carry, basis, premium, dispersion ----------------
    alias = {
        "mkt_funding_z": "funding_z",
        "mkt_funding_cum7d": "funding_cum_7d",
        "mkt_funding_persist": "funding_sign_persist",
        "mkt_basis_z": "basis_z",
        "mkt_basis_abs": "basis_abs",
        "mkt_premium_vol": "premium_vol",
        "mkt_premium_above0": "premium_time_above_zero",
        "mkt_corr_btc": "corr_btc_8h_90d",
    }
    for tag, col in alias.items():
        if col in M.columns:
            S[tag] = M[col]
    for c in ("btc_resid_lag", "eth_resid_lag", "term_basis_slope"):
        if c in M.columns:
            S[f"ctx_{c}"] = M[c]
    # LEAK, CAUGHT AND FIXED. `target` at t_obs IS the forward return, so the
    # cross-sectional std of it at t is a FORWARD quantity. Correlating that
    # with the forward market return correlates two overlapping forwards and
    # scored IC +0.1151 at t +8.71 -- comfortably the "best" signal in the
    # first run, and entirely spurious. Only PAST dispersion is admissible.
    _disp = T.groupby("t_obs")["target"].std().reindex(rm.index)
    S["dispersion_lag1"] = _disp.shift(1)
    S["dispersion_30d"] = _disp.rolling(90, min_periods=45).mean().shift(1)

    periods = {
        "train": t < splits.TRUE_HOLDOUT2_MS,
        "2026": (t >= splits.TRUE_HOLDOUT2_MS) & (t < CAP),
    }
    print("DIRECTIONAL IC -- signal at t vs FORWARD MARKET RETURN")
    print(f"  breadth {BREADTH} -> IC needed for IR 1.0:  "
          + "  ".join(f"{k} {BREADTH*BARS_YR/(HORIZONS[k]):.0f}bets "
                      f"IC>{1.0/np.sqrt(BREADTH*BARS_YR/HORIZONS[k]):.3f}"
                      for k in HORIZONS))
    print("  non-overlapping sampling; both columns are SPENT data\n")

    res = {}
    for hname, k in HORIZONS.items():
        fwd = pd.Series(
            np.expm1(np.log1p(rm).rolling(k).sum().shift(-(k - 1)).to_numpy()),
            index=rm.index)
        bar = 1.0 / np.sqrt(BREADTH * BARS_YR / k)
        per_year = BARS_YR / k                    # non-overlapping samples/yr
        print(f"=== {hname}  (need |IC| > {bar:.3f})".ljust(90, "="))
        print(f"  {'signal':<22} {'train IC':>9} {'t':>6} {'Sharpe':>8} "
              f"{'tail':>6} | {'2026 IC':>8} {'t':>6} {'Sharpe':>8} {'tail':>6}")
        rows = {}
        for sig in S.columns:
            out = {}
            line = f"  {sig:<22}"
            for pname, mask in periods.items():
                idx = np.where(mask)[0][::k]                  # non-overlapping
                r = ic_t(S[sig].to_numpy()[idx], fwd.to_numpy()[idx], per_year)
                out[pname] = r
                # `tail` flags a correlation carried by a few extreme bars.
                bang = "!" if np.isfinite(r["tail_share"]) and \
                    r["tail_share"] > 0.35 else " "
                if pname == "train":
                    flag = "*" if abs(r["ic"]) > bar else " "
                    line += (f" {r['ic']:>+9.4f} {r['t']:>+6.2f} "
                             f"{r['sharpe']:>+8.2f} {r['tail_share']:>5.0%}"
                             f"{bang}{flag}|")
                else:
                    line += (f" {r['ic']:>+8.4f} {r['t']:>+6.2f} "
                             f"{r['sharpe']:>+8.2f} {r['tail_share']:>5.0%}{bang}")
            rows[sig] = out
            print(line)
        res[hname] = {"bar": bar, "signals": rows}
        best = max(rows, key=lambda s: abs(rows[s]["train"]["ic"])
                   if np.isfinite(rows[s]["train"]["ic"]) else 0)
        b = rows[best]["train"]["ic"]
        print(f"  best |IC| on train: {best} at {b:+.4f} "
              f"({'CLEARS' if abs(b) > bar else 'below'} the {bar:.3f} bar)\n")

    print("=" * 74)
    any_clear = False
    for h, d in res.items():
        cl = [s for s, v in d["signals"].items()
              if np.isfinite(v["train"]["ic"]) and abs(v["train"]["ic"]) > d["bar"]
              and np.isfinite(v["2026"]["ic"])
              and np.sign(v["2026"]["ic"]) == np.sign(v["train"]["ic"])]
        any_clear |= bool(cl)
        print(f"  {h:<7} clears the bar AND keeps its sign in 2026: "
              + (", ".join(cl) if cl else "none"))
    print()
    if any_clear:
        print("  Something survives both filters. Worth a design pass -- but a")
        print("  sign that agrees across two SPENT periods is not evidence,")
        print("  it is the minimum bar for being worth forward-testing.")
    else:
        print("  NOTHING clears the breadth bar with a consistent sign.")
        print("  The directional book cannot reach IR 1.0 on these signals at")
        print("  these horizons. That is the go/no-go answer, and it is no.")

    out = DATA_DIR / args.out
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
