"""Purged walk-forward folds inside TRAIN. The single source of truth.

Fixed BEFORE Stage 6 runs, per docs/FEATURE_SELECTION.md §8: "How many folds,
and their length. Needs to be fixed before Stage 6 runs, not tuned once results
are visible." Changing any constant here after a target-touching stage has run
is a re-run and must be counted.

THE NUMBERS, AND WHY
--------------------
TRAIN holds 4,680 8h decision instants, but only **4,558** of them have a
defined target -- the beta/idio-vol warm-up consumes the start and the forward
window runs past the data at the end. The folds are sized on the 4,558 that can
actually be scored, not on the grid.

  (Recorded because the first version of this file was sized on 4,680 and
  raised on the first call. The correction was made BEFORE any target-touching
  number was computed -- Stage 6 aborted in `make_folds` -- so nothing was seen
  and nothing is counted. Had a result already been produced, changing these
  constants would have been a re-run.)

    MIN_TRAIN_BARS = 1803   ~600 days. The longest declared feature window is
                            90 days (270 bars), so the first model still sees
                            ~1.5 years beyond the warm-up of its slowest input.
    N_FOLDS        = 5      Enough that "clears the bar in most folds" is a
                            meaningful statement. With 3 the stability
                            criterion in §5 degenerates to a coin flip; with 10
                            the test blocks fall below a market cycle.
    TEST_BARS      = 551    ~184 days, ~6 months. 1803 + 5 x 551 = 4558
                            exactly, so the folds tile the scorable panel with
                            nothing discarded.

EXPANDING, NOT ROLLING. Each fold trains on everything before its test block.
A rolling window would hold the training size fixed and throw away history,
which at 65 bars per feature is the more expensive error.

THE PURGE
---------
PURGE_BARS = 2. The target at t_obs spans [t_obs + lag, t_obs + lag + h] =
[t+1h, t+9h], so a training row within 9h of the test block's start has a label
that overlaps the test period. ceil(9h / 8h) = 2 bars, dropped from the END of
each training block.

No embargo is needed on the other side: the folds are walk-forward, so training
is always strictly BEFORE test, and nothing after the test block enters
training at all.

WITHIN-TRAIN OVERLAP IS ZERO, AND THAT IS A PROPERTY OF THE TARGET DESIGN.
Consecutive bars span [t+1h, t+9h] and [t+9h, t+17h] -- adjacent, disjoint. So
the purge is the only correction required, and per-bar statistics need no
Newey-West adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mft import splits

MIN_TRAIN_BARS = 1803
N_FOLDS = 5
TEST_BARS = 551
PURGE_BARS = 2          # ceil((h + lag) / 8h) = ceil(9h / 8h)


@dataclass(frozen=True)
class Fold:
    k: int
    train: np.ndarray       # t_obs values, purged
    test: np.ndarray        # t_obs values

    def __repr__(self) -> str:
        return (f"Fold({self.k}: train {len(self.train)} bars to "
                f"{splits._fmt(int(self.train[-1]))}, test {len(self.test)} "
                f"bars {splits._fmt(int(self.test[0]))} .. "
                f"{splits._fmt(int(self.test[-1]))})")


def make_folds(t_obs: np.ndarray) -> list[Fold]:
    """Purged expanding walk-forward folds over TRAIN instants only.

    Raises if any instant reaches a holdout -- the folds exist to partition
    TRAIN, and a holdout leaking in here would be silent.
    """
    t = np.sort(np.asarray(t_obs, dtype=np.int64))
    if t.size and int(t[-1]) >= splits.HOLDOUT1_START_MS:
        raise splits.HoldoutLeak(
            f"make_folds received an instant at or past "
            f"{splits._fmt(splits.HOLDOUT1_START_MS)}; folds partition TRAIN only")

    need = MIN_TRAIN_BARS + N_FOLDS * TEST_BARS
    if len(t) < need:
        raise ValueError(f"need >= {need} bars for {N_FOLDS} folds, got {len(t)}")
    if len(t) != need:
        # Loud on purpose. The constants above are sized to the scorable panel
        # exactly; a different count means the panel moved and the tiling is no
        # longer the one that was pre-registered.
        print(f"  [folds] NOTE: {len(t)} bars supplied, {need} tiled, "
              f"{len(t) - need} trailing bars unused")

    folds = []
    for k in range(N_FOLDS):
        lo = MIN_TRAIN_BARS + k * TEST_BARS
        hi = lo + TEST_BARS
        train = t[: max(0, lo - PURGE_BARS)]
        folds.append(Fold(k=k, train=train, test=t[lo:hi]))
    return folds


def describe(folds: list[Fold]) -> str:
    lines = [f"{N_FOLDS} purged expanding walk-forward folds "
             f"(min train {MIN_TRAIN_BARS}, test {TEST_BARS}, "
             f"purge {PURGE_BARS})"]
    for f in folds:
        lines.append("  " + repr(f))
    return "\n".join(lines)
