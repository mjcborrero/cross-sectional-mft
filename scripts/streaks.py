"""Monthly loss streaks -- the "no two consecutive losing months" screen.

WHY THIS METRIC IS NOT A SHARPE METRIC
---------------------------------------
Under i.i.d. monthly returns the probability of a losing month is
Phi(-S/sqrt(12)) for annualised Sharpe S, and the chance of surviving n months
with no back-to-back loss follows the recursion

    W_n = q*(W_{n-1} + L_{n-1}),   L_n = p*W_{n-1},   a_n = W_n + L_n

which decays geometrically. At Sharpe 2.0 a single clean YEAR has probability
~0.47; three years ~0.10. So the screen cannot be passed by raising Sharpe to
any plausible level. It is passed by SERIAL INDEPENDENCE of monthly P&L --
losses that are single-month shocks rather than regime drawdowns.

That makes it a test of holding period and edge type, not of edge size.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits
from mft.paths import DATA_DIR

CAP = int(pd.Timestamp("2026-07-31", tz="UTC").timestamp() * 1000)
LOOKBACKS = [15, 30, 45, 60, 75, 90, 120, 150, 180, 270, 360]


def streaks(r: pd.Series) -> dict:
    """r indexed by month. Longest run of consecutive negative months."""
    neg = (r < 0).to_numpy()
    best = cur = 0
    for x in neg:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    pairs = int((neg[:-1] & neg[1:]).sum()) if len(neg) > 1 else 0
    # autocorrelation of monthly P&L: the thing the screen really tests
    ac = float(pd.Series(r.to_numpy()).autocorr(1)) if len(r) > 3 else np.nan
    return {"months": len(r), "neg": int(neg.sum()), "win%": 1 - neg.mean(),
            "longest_loss_streak": best, "back_to_back_pairs": pairs, "ac1": ac}


def theory(sharpe: float, n: int = 12) -> float:
    """P(no two consecutive losing months in n) for i.i.d. monthly returns."""
    p = float(sps.norm.cdf(-sharpe / np.sqrt(12.0)))
    q = 1 - p
    W, L = q, p
    for _ in range(n - 1):
        W, L = q * (W + L), p * W
    return W + L


def main() -> int:
    print("THE SCREEN, IN THEORY -- i.i.d. monthly returns")
    print("  P(no two consecutive losing months) purely from Sharpe:\n")
    print(f"  {'Sharpe':>7} {'P(mo loss)':>11} {'1 year':>9} {'3 years':>9} "
          f"{'5 years':>9}")
    for s in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        p = float(sps.norm.cdf(-s / np.sqrt(12.0)))
        print(f"  {s:>7.1f} {p:>11.1%} {theory(s,12):>9.1%} "
              f"{theory(s,36):>9.1%} {theory(s,60):>9.1%}")
    print("\n  Read the 3-year column: even Sharpe 3.0 fails this screen more")
    print("  often than it passes. It is not a Sharpe problem.\n")

    print("=" * 78)
    print("THE MARKET-NEUTRAL BOOK (measured, from the saved monthly reports)")
    print("=" * 78)
    print(f"  {'period':<10} {'mo':>4} {'neg':>4} {'win%':>7} "
          f"{'longest loss run':>18} {'b2b pairs':>10} {'AC(1)':>7}")
    allm = []
    for tag, f in [("train", "strategy_train"), ("2025", "strategy_holdout1"),
                   ("2026", "strategy_holdout2")]:
        d = json.loads((DATA_DIR / f"features/{f}.json").read_text())
        r = pd.Series({m["month"]: m["return"] for m in d["monthly"]})
        allm.append(r)
        s = streaks(r)
        print(f"  {tag:<10} {s['months']:>4} {s['neg']:>4} {s['win%']:>7.0%} "
              f"{s['longest_loss_streak']:>18} {s['back_to_back_pairs']:>10} "
              f"{s['ac1']:>+7.2f}")
    full = pd.concat(allm)
    s = streaks(full)
    print(f"  {'ALL':<10} {s['months']:>4} {s['neg']:>4} {s['win%']:>7.0%} "
          f"{s['longest_loss_streak']:>18} {s['back_to_back_pairs']:>10} "
          f"{s['ac1']:>+7.2f}")
    worst = full[full < 0]
    print(f"\n  worst months: "
          + ", ".join(f"{i} {v:+.1%}" for i, v in worst.nsmallest(4).items()))

    print("\n" + "=" * 78)
    print("TREND FOLLOWING (blend of 11 lookbacks) -- same screen")
    print("=" * 78)
    T = pd.read_parquet(DATA_DIR / "grid/target_8h_lag60.parquet")
    rm = T.groupby("t_obs")["fwd_rm"].first().sort_index()
    lr = np.log1p(rm)
    pos = sum(np.sign(lr.rolling(w, min_periods=w // 2).sum().shift(1))
              for w in LOOKBACKS) / len(LOOKBACKS)
    idx = pd.to_datetime(rm.index, unit="ms")
    for name, ret in [("trend blend", pd.Series((pos * rm).to_numpy(), index=idx)),
                      ("buy & hold", pd.Series(rm.to_numpy(), index=idx))]:
        m = ret.groupby(ret.index.to_period("M")).apply(lambda g: (1 + g).prod() - 1)
        m = m[m.index < pd.Period("2026-08")]
        s = streaks(m)
        print(f"  {name:<14} {s['months']:>4}mo  win {s['win%']:>4.0%}  "
              f"longest loss run {s['longest_loss_streak']:>2}  "
              f"b2b pairs {s['back_to_back_pairs']:>2}  AC(1) {s['ac1']:>+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
