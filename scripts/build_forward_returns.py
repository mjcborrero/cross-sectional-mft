"""Step 7 -- forward returns over [t_fill, t_fill+h], and the matching index leg.

Implements Step 7 of docs/TARGET_DESIGN.md. Produces the label's forward legs:

    fwd_ret    coin return from t_fill to t_fill+h
    fwd_rm     volume-weighted index return over the SAME window
    px_fill    entry price, and the close_time of the kline it came from

The window starts at t_fill = t_obs + lag, not at t_obs. That is the whole
point of the lag parameter: a target measured from t_obs claims an entry price
available at the instant of the decision, which no pipeline can achieve. Every
estimated input (beta, sigma_eps) still cuts off at t_obs.

Weights for the index leg are those stamped at t_obs -- known before the
window opens at t_obs+lag, so point-in-time holds. This makes the forward
index at t_obs identical to the Step 2 index stamped one step later, which
Gate 7 asserts rather than assumes.

Purge is h + lag from t_obs (9h at the defaults), because the label window
closes that far past the decision instant.

    python scripts/build_forward_returns.py
    python scripts/build_forward_returns.py --lags 5,15,30,60,120   # for Step 7L
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
from mft.data.store import ParquetStore
from mft.paths import DATA_DIR, NORMALIZED_DIR

MS_MIN = 60_000
MS_HOUR = 3_600_000
TOL_MS = 5 * MS_MIN
H = 8
FAILURES: list[str] = []


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def asof(ct: np.ndarray, px: np.ndarray, instants: np.ndarray):
    """Last close STRICTLY before each instant, plus the close_time used."""
    i = np.searchsorted(ct, instants, side="left") - 1
    out = np.full(len(instants), np.nan)
    when = np.full(len(instants), np.nan)
    ok = i >= 0
    if ok.any():
        v = i[ok]
        fresh = (instants[ok] - ct[v]) <= TOL_MS
        sel = np.where(ok)[0][fresh]
        out[sel] = px[v][fresh]
        when[sel] = ct[v][fresh]
    return out, when


def build(lag_min: int, t_obs: np.ndarray, klines: dict, wts: pd.DataFrame,
          step_ms: int) -> pd.DataFrame:
    """Forward legs for one lag."""
    lag_ms = lag_min * MS_MIN
    fill_at = t_obs + lag_ms

    px, when = {}, {}
    for sym, (ct, p) in klines.items():
        px[sym], when[sym] = asof(ct, p, fill_at)
    P = pd.DataFrame(px, index=t_obs)
    Wt = pd.DataFrame(when, index=t_obs)

    # Forward return: next grid step's fill price over this one's. Only when
    # the next observation is exactly one step later -- a gap must never be
    # booked as an 8h return.
    nxt = pd.Series(t_obs, index=t_obs).shift(-1)
    consec = (nxt - pd.Series(t_obs, index=t_obs)) == step_ms
    F = (P.shift(-1) / P - 1.0).where(consec, np.nan)

    # Index leg over the same window, weights stamped at t_obs (they predate
    # the window, which opens at t_obs+lag).
    wq = wts.pivot(index="t_obs", columns="symbol", values="weight").reindex(t_obs)
    wq = wq.reindex(columns=F.columns)
    M = wq.where(F.notna())
    M = M.div(M.sum(axis=1), axis=0)
    rm = (M * F).sum(axis=1, min_count=1).where(M.notna().sum(axis=1) >= 10)

    long = F.stack().rename("fwd_ret").reset_index()
    long.columns = ["t_obs", "symbol", "fwd_ret"]
    long["px_fill"] = P.stack().reindex(
        pd.MultiIndex.from_frame(long[["t_obs", "symbol"]])).to_numpy()
    long["px_close_time"] = Wt.stack().reindex(
        pd.MultiIndex.from_frame(long[["t_obs", "symbol"]])).to_numpy()
    long["fwd_rm"] = long["t_obs"].map(rm)
    long["lag_min"] = lag_min
    return long.dropna(subset=["fwd_ret", "fwd_rm", "px_fill"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="grid/decision_grid_8h.parquet")
    ap.add_argument("--weights", default="grid/market_weights_30d.parquet")
    ap.add_argument("--lags", default="60", help="comma list of lags in minutes")
    ap.add_argument("--primary", type=int, default=60, help="lag that is gated and written")
    ap.add_argument("--h", type=int, default=H)
    # The Step 2 index to cross-check against MUST be built on the same grid
    # and horizon. Hardcoding the 8h one made the check compare a 24h forward
    # index against an 8h index -- a real mismatch, but in the CHECK.
    ap.add_argument("--ref-index", default=None,
                    help="Step 2 index to cross-check against "
                         "(default: grid/market_index_{h}h.parquet)")
    args = ap.parse_args()

    lags = [int(x) for x in args.lags.split(",")]
    step_ms = args.h * MS_HOUR
    wall = splits.HOLDOUT1_START_MS

    dec = pd.read_parquet(DATA_DIR / args.grid)
    wts = pd.read_parquet(DATA_DIR / args.weights)
    t_obs = np.sort(dec["t_obs"].unique())
    universe = sorted(dec["symbol"].unique())

    print(f"Step 7 -- forward returns  h={args.h}h  lags={lags} (primary {args.primary})")
    print(f"  {len(t_obs):,} decision instants, {len(universe)} coins\n")

    store = ParquetStore(NORMALIZED_DIR)
    klines = {}
    for sym in universe:
        kl = store.load("perp_klines", sym, columns=["close_time", "close"])
        kl = kl[kl["close_time"] < wall].sort_values("close_time")
        klines[sym] = (kl["close_time"].to_numpy(), kl["close"].to_numpy())

    frames = {lag: build(lag, t_obs, klines, wts, step_ms) for lag in lags}
    df = frames[args.primary]
    lag_ms = args.primary * MS_MIN

    # ---- GATE 7 -----------------------------------------------------------
    print("GATE 7\n" + "=" * 74)

    # Entry price must come from a candle that CLOSED before the fill instant.
    fill_at = df["t_obs"] + lag_ms
    gate(bool((df["px_close_time"] < fill_at).all()),
         "entry price closed strictly before t_fill (no lookahead at entry)",
         f"max age {int((fill_at - df.px_close_time).max()) / MS_MIN:.1f} min")

    # The whole label window must resolve before the holdout wall.
    win_end = df["t_obs"] + lag_ms + step_ms
    gate(bool((win_end <= wall).all()),
         f"label window closes before the holdout wall (purge = "
         f"{args.h}h + {args.primary}min)",
         f"last close {splits._fmt(int(win_end.max()))}")

    purge_h = args.h + args.primary / 60.0
    try:
        splits.assert_train_only(df.assign(bar_close_time=df["t_obs"]))
        gate(True, "splits.assert_train_only(t_obs)")
    except splits.HoldoutLeak as e:
        gate(False, "splits.assert_train_only(t_obs)", str(e)[:70])

    # Consistency: the forward index at t must equal the Step 2 index stamped
    # one step later. Two independent constructions of the same quantity.
    ref_path = args.ref_index or f"grid/market_index_{args.h}h.parquet"
    idx2 = pd.read_parquet(DATA_DIR / ref_path)
    ref = idx2.set_index("t_obs")["r_m"]
    chk = df.drop_duplicates("t_obs")[["t_obs", "fwd_rm"]].copy()
    chk["ref"] = (chk["t_obs"] + step_ms).map(ref)
    chk = chk.dropna()
    dev = float((chk["fwd_rm"] - chk["ref"]).abs().max())
    gate(dev < 1e-12,
         "forward index == Step 2 index shifted one step (independent agreement)",
         f"max |diff| {dev:.2e} over {len(chk):,} steps")

    # Coverage: everything except the tail the purge removes AND the head the
    # weights warm-up removes. A forward INDEX cannot exist before the weights
    # it is built from, and that is a property of the data, not a defect.
    #
    # This matters because the warm-up is grid-dependent in a way that is easy
    # to miss: build_market_weights uses `min_periods=min_obs`, a COUNT of
    # observations. On an 8h grid 30 observations arrive in 10 days; on a daily
    # grid the same count takes 30 days. So the 8h build lost 29 bars (0.45%)
    # and the 24h build loses 87 (4.0%) for the identical reason. Same bug
    # class as the bar-count feature windows and anatomy.py's bar_duration.
    # The denominator is bars that HAVE weights, not bars at or after the first
    # one. On the 24h grid the weights also blank out for 29 days after each of
    # the two 2022 archive gaps -- the same min_obs-as-a-count effect, since a
    # daily grid needs 29 bars to re-accumulate 30 observations where an 8h grid
    # needs ~10. Two contiguous runs, 2022-03-02..03-30 and 2022-04-04..05-02,
    # 58 bars, fully explained.
    w_have = np.unique(wts["t_obs"].to_numpy())
    usable = t_obs[(t_obs + lag_ms + step_ms <= wall) & np.isin(t_obs, w_have)]
    cov = df["t_obs"].nunique() / len(usable)
    gate(cov > 0.99, "coverage of usable decision instants > 99%",
         f"{df.t_obs.nunique():,} / {len(usable):,} = {cov:.3%}")

    gate(bool(np.isfinite(df[["fwd_ret", "fwd_rm"]]).all().all()),
         "no NaN or inf in forward legs")

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"GATE 7 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1

    # ---- census -----------------------------------------------------------
    print("\nCENSUS (primary lag)\n" + "=" * 74)
    ann = np.sqrt(365 * 24 / args.h)
    print(f"  {len(df):,} rows   {df.symbol.nunique()} coins   "
          f"{df.t_obs.nunique():,} cross-sections")
    print(f"  {splits._fmt(df.t_obs.min())} -> {splits._fmt(df.t_obs.max())}")
    print(f"  fwd_ret  std {df.fwd_ret.std():.5f} -> {df.fwd_ret.std()*ann:.1%} annualised")
    print(f"  fwd_rm   std {df.fwd_rm.std():.5f} -> {df.fwd_rm.std()*ann:.1%} annualised")
    print(f"  purge applied: {purge_h:.2f}h from t_obs")

    if len(lags) > 1:
        print("\n  forward-leg summary by lag:")
        print(f"    {'lag':>6} {'rows':>9} {'std fwd_ret':>12} {'std fwd_rm':>11}")
        for lag in lags:
            f = frames[lag]
            print(f"    {lag:>4}m {len(f):>9,} {f.fwd_ret.std():>12.5f} "
                  f"{f.fwd_rm.std():>11.5f}")

    # Was hardcoded to "forward_8h_..." regardless of --h, so a 24h build
    # silently OVERWROTE the 8h artifact. The horizon must be in the name.
    out = DATA_DIR / f"grid/forward_{args.h}h_lag{args.primary}.parquet"
    df[["symbol", "t_obs", "px_fill", "fwd_ret", "fwd_rm"]].to_parquet(out, index=False)
    out.with_suffix(".manifest.json").write_text(json.dumps({
        "step": 7, "h_hours": args.h, "lag_min": args.primary,
        "purge_hours": purge_h,
        "window": "[t_obs+lag, t_obs+lag+h]",
        "index_weights_stamped_at": "t_obs (predates window open)",
        "rows": int(len(df)), "cross_sections": int(df.t_obs.nunique()),
        "coverage_of_usable": round(float(cov), 6),
        "fwd_ret_std": round(float(df.fwd_ret.std()), 8),
        "fwd_rm_std": round(float(df.fwd_rm.std()), 8),
        "index_agreement_max_abs_diff": dev,
    }, indent=2), encoding="utf-8")

    if len(lags) > 1:
        allf = pd.concat(frames.values(), ignore_index=True)
        allf[["symbol", "t_obs", "lag_min", "px_fill", "fwd_ret", "fwd_rm"]].to_parquet(
            DATA_DIR / f"grid/forward_{args.h}h_lagsweep.parquet", index=False)
        print(f"  wrote lag sweep -> grid/forward_{args.h}h_lagsweep.parquet")

    print("\n" + "=" * 74)
    print(f"GATE 7 PASSED. Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
