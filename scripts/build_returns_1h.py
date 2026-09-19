"""Build the 1h return panel and 1h market index (input to Steps 5 and 6).

Step 4 chose 1h as the beta-estimation frequency. Beta (Step 5) and
idiosyncratic volatility (Step 6) are both estimated from these returns, so
they are built once here rather than recomputed per step.

Same conventions as the decision grid:
  - price at instant T = last 1m close STRICTLY before T, stale > 5min = missing
  - never forward-filled; a missing price yields a missing return
  - index uses weights asof-matched to strictly BEFORE the return window opens
  - train only: everything ends at the holdout wall

    python scripts/build_returns_1h.py
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
MIN_COINS = 10


def asof_px(ct: np.ndarray, px: np.ndarray, instants: np.ndarray) -> np.ndarray:
    i = np.searchsorted(ct, instants, side="left") - 1
    out = np.full(len(instants), np.nan)
    ok = i >= 0
    if ok.any():
        v = i[ok]
        fresh = (instants[ok] - ct[v]) <= TOL_MS
        out[np.where(ok)[0][fresh]] = px[v][fresh]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-09-01")
    ap.add_argument("--weights", default="grid/market_weights_30d.parquet")
    ap.add_argument("--out", default="grid/returns_1h.parquet")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    universe = [l.strip().upper() for l in
                (repo / "config/universe.txt").read_text(encoding="utf-8-sig").splitlines()
                if l.strip() and not l.startswith("#")]

    wall = splits.HOLDOUT1_START_MS
    lo = int(pd.Timestamp(args.start, tz="UTC").value // 10**6)
    grid = np.arange(-(-lo // MS_HOUR) * MS_HOUR, wall, MS_HOUR, dtype=np.int64)

    print(f"1h return panel: {args.start} -> {splits._fmt(wall)}  ({len(grid):,} hours)")

    store = ParquetStore(NORMALIZED_DIR)
    px = {}
    for sym in universe:
        kl = store.load("perp_klines", sym, columns=["close_time", "close"])
        kl = kl[kl["close_time"] < wall].sort_values("close_time")
        px[sym] = asof_px(kl["close_time"].to_numpy(), kl["close"].to_numpy(), grid)

    P = pd.DataFrame(px, index=grid)
    R = P.pct_change().replace([np.inf, -np.inf], np.nan)
    R.index.name = "t"

    # Weights in force strictly before each return window opens.
    wts = pd.read_parquet(DATA_DIR / args.weights)
    wq = wts.pivot(index="t_obs", columns="symbol", values="weight")
    W = wq.reindex(wq.index.union(grid)).ffill().reindex(grid).shift(1)
    W = W.reindex(columns=R.columns)

    M = W.where(R.notna())
    M = M.div(M.sum(axis=1), axis=0)
    r_m = (M * R).sum(axis=1, min_count=1)
    r_m = r_m.where(M.notna().sum(axis=1) >= MIN_COINS)

    r_m.index = pd.Index(grid, name="t")
    idx_df = pd.DataFrame({"t": grid, "r_m": r_m.to_numpy()})

    long = R.stack().rename("ret").reset_index()
    long.columns = ["t", "symbol", "ret"]
    long = long.merge(idx_df, on="t", how="left")
    long = long.dropna(subset=["ret", "r_m"])

    print(f"  {len(long):,} (symbol, hour) returns   "
          f"{long.symbol.nunique()} symbols   {r_m.notna().sum():,} index hours")
    print(f"  r_m 1h vol {r_m.std():.5f}  -> annualised {r_m.std()*np.sqrt(365*24):.1%}")

    assert long["t"].max() < wall, "1h panel crosses the holdout wall"

    out = DATA_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out, index=False)
    out.with_suffix(".manifest.json").write_text(json.dumps({
        "frequency": "1h", "start": args.start,
        "rows": int(len(long)), "symbols": int(long.symbol.nunique()),
        "index_hours": int(r_m.notna().sum()),
        "r_m_vol_1h": round(float(r_m.std()), 8),
        "staleness_tolerance_min": TOL_MS // MS_MIN,
        "min_coins_for_index": MIN_COINS,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
