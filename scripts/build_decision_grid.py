"""Step 0 -- price foundation on the 8h decision grid.

Implements Step 0 of docs/TARGET_DESIGN.md. Resamples 1-minute perp klines onto
the funding-aligned decision grid and produces the price panel every later step
builds on. No bar construction, no features (see TARGET_DESIGN.md sec 4).

    t_obs   in {00:00, 08:00, 16:00} UTC   -- decision instant, data cutoff
    t_fill  = t_obs + lag                  -- when the position is established

Two prices per row, both taken as "last kline that CLOSED strictly before the
instant":

    px_obs   price observable at t_obs   (what the model may condition on)
    px_fill  price at t_fill             (what the position is actually entered at)

Because grid points are h apart and h is the holding period, the exit price for
row t is simply the px_fill of the NEXT row for that symbol. Step 7 uses that;
this script only lays the prices down.

GATES (docs/TARGET_DESIGN.md, Gate 0). All are hard: the script ABORTS and
writes nothing if any fails. A gate that prints instead of stopping is not a
gate.

Missing data is never forward-filled. If the newest kline before an instant is
staler than --tolerance, that (symbol, t_obs) is dropped outright, and any
cross-section left with fewer than --min-symbols coins is dropped whole.
Forward-filling would fabricate a zero return while the rest of the universe
moved, which ranks the coin mid-pack on invented data and then dumps the whole
accumulated jump into the next return.

    python scripts/build_decision_grid.py
    python scripts/build_decision_grid.py --lag 15 --out grid/decision_grid_8h_lag15.parquet
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

# A listed coin missing more than this share of its own span is a data-quality
# problem worth stopping for, not a rounding error. Stated here, before the run.
MAX_GAP_FRAC = 0.01

FAILURES: list[str] = []


def gate(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def fmt(ms) -> str:
    return splits._fmt(int(ms))


def load_universe(path: Path) -> list[str]:
    syms = [ln.strip().upper() for ln in path.read_text(encoding="utf-8-sig").splitlines()]
    return [s for s in syms if s and not s.startswith("#")]


def build_grid(start_ms: int, end_ms: int, h_hours: int) -> np.ndarray:
    """Grid instants at multiples of h hours from the UTC epoch, in [start, end).

    h divides 24 for h in {8, 24}, so the grid lands on 00:00/08:00/16:00 UTC --
    the Binance funding timestamps -- with no drift.
    """
    step = h_hours * MS_HOUR
    first = -(-start_ms // step) * step  # ceil to the next grid point
    return np.arange(first, end_ms, step, dtype=np.int64)


def asof_prices(kl: pd.DataFrame, instants: np.ndarray,
                tolerance_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Last close strictly before each instant, plus that quote's age in ms.

    Returns (price, age). Price is NaN where no kline exists within tolerance --
    never carried forward.
    """
    ct = kl["close_time"].to_numpy()
    px = kl["close"].to_numpy()
    # searchsorted 'left' gives the count of close_times < instant, so idx-1 is
    # the newest kline that closed strictly before it.
    idx = np.searchsorted(ct, instants, side="left") - 1
    out = np.full(len(instants), np.nan)
    age = np.full(len(instants), np.nan)
    valid = idx >= 0
    if valid.any():
        v = idx[valid]
        a = instants[valid] - ct[v]
        p = px[v]
        fresh = a <= tolerance_ms
        sel = np.where(valid)[0][fresh]
        out[sel] = p[fresh]
        age[sel] = a[fresh]
    return out, age


def runs(instants: np.ndarray, step_ms: int) -> list[tuple[int, int, int]]:
    """Collapse sorted instants into contiguous (start, end, count) runs."""
    if len(instants) == 0:
        return []
    out, s, prev, n = [], int(instants[0]), int(instants[0]), 1
    for x in instants[1:]:
        x = int(x)
        if x == prev + step_ms:
            prev, n = x, n + 1
        else:
            out.append((s, prev, n))
            s, prev, n = x, x, 1
    out.append((s, prev, n))
    return out


def window_volume(kl: pd.DataFrame, instants: np.ndarray, span_ms: int) -> np.ndarray:
    """Quote volume summed over [instant - span, instant) -- trailing, no peek."""
    ct = kl["close_time"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(kl["quote_volume"].to_numpy(dtype="float64"))])
    hi = np.searchsorted(ct, instants, side="left")
    lo = np.searchsorted(ct, instants - span_ms, side="left")
    return cum[hi] - cum[lo]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=8, help="holding period, hours (default 8)")
    ap.add_argument("--lag", type=int, default=60, help="execution lag, minutes (default 60)")
    ap.add_argument("--tolerance", type=int, default=5,
                    help="max staleness of a quote, minutes (default 5)")
    ap.add_argument("--min-symbols", type=int, default=15,
                    help="minimum coins for a usable cross-section (default 15)")
    ap.add_argument("--universe", default="config/universe.txt")
    ap.add_argument("--out", default=None, help="output path relative to DATA_DIR")
    args = ap.parse_args()

    h_ms = args.h * MS_HOUR
    lag_ms = args.lag * MS_MIN
    tol_ms = args.tolerance * MS_MIN
    wall = splits.HOLDOUT1_START_MS

    repo = Path(__file__).resolve().parent.parent
    universe = load_universe(repo / args.universe)
    out_rel = args.out or f"grid/decision_grid_{args.h}h.parquet"
    out_path = DATA_DIR / out_rel

    print(f"Step 0 -- decision grid  h={args.h}h  lag={args.lag}min  "
          f"tolerance={args.tolerance}min  min_symbols={args.min_symbols}")
    print(f"Universe: {len(universe)} symbols from {args.universe}")
    print(f"Holdout wall: {fmt(wall)}  (train only; every price must precede it)\n")

    store = ParquetStore(NORMALIZED_DIR)

    # Start the grid at the first kline in the universe, not at the epoch, so
    # "missing" means a real gap rather than "this coin did not exist yet".
    starts = []
    for sym in universe:
        if store.files("perp_klines", sym):
            c = store.load("perp_klines", sym, columns=["close_time"])["close_time"]
            if len(c):
                starts.append(int(c.min()))
    if not starts:
        raise SystemExit("No perp_klines found for any universe symbol.")

    # Grid runs while the FILL instant still precedes the wall. Rows whose full
    # forward window also fits are selected in Step 7 -- the last row here is
    # needed as the exit price for the last labelable row.
    grid = build_grid(min(starts), wall - lag_ms, args.h)
    step_ms = args.h * MS_HOUR
    print(f"Grid: {len(grid):,} instants  {fmt(grid[0])} -> {fmt(grid[-1])}")

    frames = []
    missing_syms = []
    per_symbol = []

    for sym in universe:
        if not store.files("perp_klines", sym):
            missing_syms.append(sym)
            continue
        kl = store.load("perp_klines", sym, columns=["close_time", "close", "quote_volume"])
        kl = kl[kl["close_time"] < wall]
        if kl.empty:
            missing_syms.append(sym)
            continue
        kl = kl.sort_values("close_time", kind="mergesort").reset_index(drop=True)

        px_obs, age_obs = asof_prices(kl, grid, tol_ms)
        px_fill, age_fill = asof_prices(kl, grid + lag_ms, tol_ms)
        qv = window_volume(kl, grid, h_ms)

        df = pd.DataFrame({
            "symbol": sym,
            "t_obs": grid,
            "t_fill": grid + lag_ms,
            "px_obs": px_obs,
            "px_fill": px_fill,
            "age_obs_ms": age_obs,
            "age_fill_ms": age_fill,
            "qv_window": qv,
        })
        ok = df["px_obs"].notna() & df["px_fill"].notna()
        if not ok.any():
            missing_syms.append(sym)
            continue

        # Separate "not listed yet" from "listed, but data missing". Only the
        # second is a data-quality problem.
        first_v, last_v = int(df.loc[ok, "t_obs"].min()), int(df.loc[ok, "t_obs"].max())
        live = df[(df["t_obs"] >= first_v) & (df["t_obs"] <= last_v)]
        live_ok = live["px_obs"].notna() & live["px_fill"].notna()
        gaps = live.loc[~live_ok, "t_obs"].to_numpy()

        frames.append(df[ok])
        per_symbol.append({
            "symbol": sym, "first": first_v, "last": last_v,
            "n_live": int(len(live)), "n_gap": int(len(gaps)),
            "gap_frac": len(gaps) / max(len(live), 1), "gaps": gaps,
        })

    if missing_syms:
        print(f"\n  MISSING from perp_klines: {missing_syms}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["t_obs", "symbol"], kind="mergesort").reset_index(drop=True)

    # ---- cross-section floor -------------------------------------------
    counts = panel.groupby("t_obs")["symbol"].size()
    thin = counts[counts < args.min_symbols]
    panel = panel[~panel["t_obs"].isin(thin.index)].reset_index(drop=True)

    # ---- GATE 0 ---------------------------------------------------------
    print(f"\nGATE 0\n" + "=" * 74)
    gate(not missing_syms, f"all {len(universe)} universe symbols present",
         f"missing: {missing_syms}" if missing_syms else f"{len(universe)}/{len(universe)}")
    gate(not panel.duplicated(["symbol", "t_obs"]).any(),
         "no duplicate (symbol, t_obs)",
         f"{int(panel.duplicated(['symbol','t_obs']).sum())} dupes")

    finite = np.isfinite(panel[["px_obs", "px_fill"]].to_numpy()).all()
    positive = (panel[["px_obs", "px_fill"]].to_numpy() > 0).all()
    gate(bool(finite and positive), "all prices finite and positive")

    # The no-forward-fill guarantee, asserted rather than assumed.
    worst_age = float(panel[["age_obs_ms", "age_fill_ms"]].to_numpy().max())
    gate(worst_age <= tol_ms, f"no quote staler than {args.tolerance}min",
         f"worst {worst_age:.0f}ms")

    # Interior gaps: instants where a LISTED coin had no data. Threshold stated
    # before the run; the observed value is reported either way.
    worst_gap = max((r["gap_frac"] for r in per_symbol), default=0.0)
    worst_sym = max(per_symbol, key=lambda r: r["gap_frac"])["symbol"] if per_symbol else "-"
    gate(worst_gap <= MAX_GAP_FRAC,
         f"no symbol missing >{MAX_GAP_FRAC:.0%} of its listed span",
         f"worst {worst_sym} {worst_gap:.2%}")

    # Off-by-one invariant: the quote must have closed strictly BEFORE its instant.
    strict_obs = bool((panel["age_obs_ms"] > 0).all())
    strict_fill = bool((panel["age_fill_ms"] > 0).all())
    gate(strict_obs and strict_fill, "every quote closed strictly before its instant")

    gate(bool((panel["t_fill"] < wall).all()),
         "every fill instant precedes the holdout wall",
         f"last {fmt(panel.t_fill.max())}")
    try:
        splits.assert_train_only(panel, time_col="t_fill")
        gate(True, "splits.assert_train_only(t_fill)")
    except splits.HoldoutLeak as e:
        gate(False, "splits.assert_train_only(t_fill)", str(e)[:70])

    final_counts = panel.groupby("t_obs")["symbol"].size()
    gate(bool((final_counts >= args.min_symbols).all()),
         f"every retained cross-section has >= {args.min_symbols} symbols",
         f"min {int(final_counts.min())}")

    # Returns must be usable: finite and non-degenerate per symbol.
    rets = panel.sort_values(["symbol", "t_obs"]).groupby("symbol")["px_fill"].pct_change()
    gate(bool(np.isfinite(rets.dropna()).all()), "all fill-to-fill returns finite")
    zero_var = [s for s, g in panel.groupby("symbol") if g["px_fill"].nunique() <= 1]
    gate(not zero_var, "no zero-variance symbol", f"{zero_var}" if zero_var else "")

    if FAILURES:
        print("\n" + "=" * 74)
        print(f"GATE 0 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        print("Nothing written. Fix the data layer before proceeding to Step 1.")
        return 1

    # ---- census (reported, not gated) -----------------------------------
    print(f"\nCENSUS\n" + "=" * 74)
    print(f"  rows {len(panel):,}   cross-sections {panel.t_obs.nunique():,}   "
          f"symbols {panel.symbol.nunique()}")
    print(f"  {fmt(panel.t_obs.min())} -> {fmt(panel.t_obs.max())}")
    kept = panel.groupby("symbol").size()
    print(f"\n  {'symbol':<10} {'kept':>7} {'listed from':<17} {'gaps':>6} {'gap %':>7}")
    for r in per_symbol:
        n = int(kept.get(r["symbol"], 0))
        print(f"  {r['symbol']:<10} {n:>7,} {fmt(r['first']):<17} "
              f"{r['n_gap']:>6,} {r['gap_frac']:>6.2%}")

    # Data gaps, made explicit rather than buried in the thin-drop count.
    gappy = [r for r in per_symbol if r["n_gap"]]
    print(f"\n  DATA GAPS (listed coin, no kline) -- {len(gappy)} of {len(per_symbol)} symbols:")
    if not gappy:
        print("    none")
    for r in sorted(gappy, key=lambda x: -x["n_gap"]):
        rr = runs(r["gaps"], step_ms)
        span = ", ".join(f"{fmt(a)} -> {fmt(b)} ({c})" for a, b, c in rr[:3])
        more = f"  (+{len(rr)-3} more)" if len(rr) > 3 else ""
        print(f"    {r['symbol']:<10} {r['n_gap']:>4} instants in {len(rr)} run(s): {span}{more}")

    print(f"\n  thin cross-sections dropped (<{args.min_symbols} symbols): {len(thin)}")
    if len(thin):
        pre = thin.index[thin.index < panel.t_obs.min()]
        interior = thin.index[thin.index >= panel.t_obs.min()]
        print(f"    {len(pre)} before the universe reached {args.min_symbols} coins "
              f"(expected: coins still listing)")
        if len(interior):
            print(f"    {len(interior)} INTERIOR -- caused by the data gaps above:")
            for a, b, c in runs(np.sort(interior.to_numpy()), step_ms):
                print(f"      {fmt(a)} -> {fmt(b)}  ({c} instants)")

    yr = panel.assign(y=pd.to_datetime(panel.t_obs, unit="ms", utc=True).dt.year)
    print("\n  cross-sections per year:")
    for y, g in yr.groupby("y"):
        print(f"    {y}  {g.t_obs.nunique():>6,}  ({g.symbol.nunique()} symbols)")

    # ---- write -----------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    manifest = {
        "step": 0,
        "h_hours": args.h,
        "lag_minutes": args.lag,
        "tolerance_minutes": args.tolerance,
        "min_symbols": args.min_symbols,
        "grid_times_utc": [f"{i*args.h:02d}:00" for i in range(24 // args.h)],
        "holdout_wall_ms": int(wall),
        "purge_note": f"labelable rows require t_fill + {args.h}h < wall (Step 7)",
        "rows": int(len(panel)),
        "cross_sections": int(panel.t_obs.nunique()),
        "symbols": sorted(panel.symbol.unique().tolist()),
        "t_obs_first_ms": int(panel.t_obs.min()),
        "t_obs_last_ms": int(panel.t_obs.max()),
        "thin_cross_sections_dropped": int(len(thin)),
        "source": "normalized/perp_klines (1m)",
        "max_gap_frac_threshold": MAX_GAP_FRAC,
        "data_gaps": {
            r["symbol"]: [
                {"from": fmt(a), "to": fmt(b), "instants": c}
                for a, b, c in runs(r["gaps"], step_ms)
            ]
            for r in per_symbol if r["n_gap"]
        },
    }
    mpath = out_path.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n" + "=" * 74)
    print(f"GATE 0 PASSED. Wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"Manifest: {mpath}")
    print("Step 1 (point-in-time volume weights) may proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
