"""Trend following on the market index: long / flat / short.

WHY TEST THIS WHEN THE IC SAID NO
----------------------------------
scripts/directional_ic.py found trend IC below the breadth bar at every horizon
(best +0.104 weekly, t +1.70 against a 0.109 bar). That is a LINEAR measure. A
sign rule is not linear -- it can work when correlation is weak if only the
extremes carry information -- and time-series momentum is the most replicated
effect in cross-asset finance. So it gets a direct test rather than dismissal
by proxy.

THE TRAP THIS IS BUILT TO AVOID
--------------------------------
The market rose 522% over train. ANY long-biased rule looks brilliant there,
and "trend following works" is the easiest false conclusion available on this
sample. So:

  * BUY AND HOLD is reported as the benchmark on every line. A trend rule that
    does not beat it has produced nothing, however good its Sharpe looks.
  * TIME LONG is reported, because a rule that is long 85% of the time in a
    bull market is a long position wearing a signal's clothes.
  * long/flat and long/short are separated. Long/flat inherits the market's
    drift; long/short does not, and the difference is where the actual timing
    skill shows up.

Positions are taken on the SIGN of trailing log return, entered at t+lag with
the same 60min execution lag as everything else, held one bar, no smoothing.
Zero fees per the design assumption; turnover is reported.

Both holdouts are spent. Neither 2025 nor 2026 is out-of-sample here.
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

CAP = int(pd.Timestamp("2026-07-31", tz="UTC").timestamp() * 1000)
BY = 1095
LOOKBACKS = {"7d": 21, "30d": 90, "90d": 270}


def stats(r: pd.Series, pos: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {}
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    eq = (1 + r).cumprod()
    out = {"sharpe": mu / sd * np.sqrt(BY) if sd else np.nan,
           "ann": float((1 + mu) ** BY - 1),
           "maxdd": float((eq / eq.cummax() - 1).min()),
           "vol": float(sd * np.sqrt(BY))}
    if pos is not None:
        p = pos.reindex(r.index)
        out["time_long"] = float((p > 0).mean())
        out["turn"] = float(p.diff().abs().mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features/momentum.json")
    args = ap.parse_args()

    T = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet")
    rm = T.groupby("t_obs")["fwd_rm"].first().sort_index()
    t = np.asarray(rm.index)
    lr = np.log1p(rm)

    periods = {
        "train": t < splits.TRUE_HOLDOUT1_MS,
        "2025": (t >= splits.TRUE_HOLDOUT1_MS) & (t < splits.TRUE_HOLDOUT2_MS),
        "2026": (t >= splits.TRUE_HOLDOUT2_MS) & (t < CAP),
    }

    print("TREND FOLLOWING ON THE MARKET INDEX")
    print("  buy&hold is the benchmark on every line; a rule that does not beat")
    print("  it has produced nothing. 'long%' exposes long-bias.\n")

    # --- benchmark -------------------------------------------------------
    print(f"  {'':<22}" + "".join(f"{p:>26}" for p in periods))
    print(f"  {'strategy':<22}" + "".join(f"{'Sharpe    ann   long%':>26}"
                                          for p in periods))
    print("  " + "-" * 100)
    res = {}
    bh = {}
    line = f"  {'BUY & HOLD':<22}"
    for pname, mask in periods.items():
        s = stats(rm[mask])
        bh[pname] = s
        line += f"{s['sharpe']:>+10.2f}{s['ann']:>9.1%}{'--':>7}"
    print(line)
    res["buy_hold"] = bh
    print()

    for mode in ("long/flat", "long/short"):
        for tag, w in LOOKBACKS.items():
            sig = lr.rolling(w, min_periods=w // 2).sum().shift(1)
            pos = np.sign(sig) if mode == "long/short" else (sig > 0).astype(float)
            pos = pd.Series(pos, index=rm.index)
            ret = pos * rm
            name = f"{mode} {tag}"
            row = {}
            line = f"  {name:<22}"
            for pname, mask in periods.items():
                s = stats(ret[mask], pos[mask])
                row[pname] = s
                if s:
                    line += (f"{s['sharpe']:>+10.2f}{s['ann']:>9.1%}"
                             f"{s['time_long']:>7.0%}")
                else:
                    line += f"{'--':>26}"
            res[name] = row
            print(line)
        print()

    # --- did any rule beat buy & hold in EVERY period? -------------------
    print("=" * 102)
    beats = []
    for name, row in res.items():
        if name == "buy_hold":
            continue
        if all(row.get(p, {}).get("sharpe", -9) > bh[p]["sharpe"] for p in periods):
            beats.append(name)
    print("  beats buy&hold in ALL THREE periods: "
          + (", ".join(beats) if beats else "NONE"))

    pos26 = [n for n, r in res.items()
             if n != "buy_hold" and r.get("2026", {}).get("sharpe", -9) > 0]
    print("  positive Sharpe in 2026: " + (", ".join(pos26) if pos26 else "NONE"))

    print("\n  Reference: the market-neutral book was +2.86 / +2.83 / -0.35.")
    out = DATA_DIR / args.out
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
