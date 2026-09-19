"""Train / holdout boundaries -- the single source of truth for the split.

Re-baselined 2026-08-04 after the research purge. The split is now:

    train     : data start (2020-01-01) .. 2024-12-31   -- ALL research happens here
    holdout1  : 2025-01-01 .. 2025-12-31                -- first out-of-sample year
    holdout2  : 2026-01-01 .. data end                  -- second out-of-sample year

Rule: fit, select, tune, and decide using `train` only. `holdout1` and
`holdout2` are read at the END of a research programme, once, to measure what
was built. Every time a holdout is looked at, it is partly spent; looking at it
repeatedly turns it back into a training set with extra steps.

Two layers enforce this, and you should use both:

  1. `paths.TRAIN_DIR` -- a materialized train-only copy of the panel written
     by scripts/materialize_train.py. Every file under it already ends at the
     purged boundary, so research pointed there cannot reach a holdout row.
     This is the layer that survives a bug in the layer below.
  2. `assert_train_only(df)` -- call it in anything that fits, tunes, or
     selects, for the cases that read DATA_DIR directly.

PURGING
-------
Labels are forward-looking, so a bar near the end of a period carries an
outcome that resolves in the NEXT period. Without purging, a model trained
through 2024-12-31 would learn from price action that happened in 2025.

Each period therefore ends `horizon` hours before its nominal boundary:

    train usable end    = 2025-01-01 00:00 UTC - horizon
    holdout1 usable end = 2026-01-01 00:00 UTC - horizon
    holdout2 usable end = last bar in the data  - horizon

`horizon` defaults to MAX_LABEL_HORIZON_H (120h), the longest label in the
ladder (fwd_return_120h / rel_target_120h). The triple-barrier label caps at
48h, so 120h dominates. If a study only ever uses the 24h target it may pass
horizon_hours=24 and recover four days of data -- but the default is the
conservative one, because a purge that is too small is silent contamination
and a purge that is too large only costs a handful of bars.

Purging each period independently (rather than only at the train boundary)
keeps holdout1 and holdout2 mutually independent: holdout1's labels resolve
inside 2025, so its result does not depend on 2026 price action.

NOTE ON WHAT THE SPLIT DOES NOT FIX
-----------------------------------
- Features may look back across a boundary (up to 90 days). That is correct
  and not leakage: at decision time that history was genuinely available.
- The 20-coin universe was picked using present-day liquidity, so it carries
  survivorship bias in every period. The split does not address this.
- 2025 and 2026 were both evaluated repeatedly under the pre-purge research
  code. They are cleaner than they were (that code and its conclusions are
  deleted) but they are not virgin data. Treat holdout1 especially as
  "partly spent" and weight holdout2 and post-2026-08 forward data higher.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pandas as pd

_H_MS = 3_600_000

# Longest forward label in the ladder. Governs the default purge.
MAX_LABEL_HORIZON_H = 120
MAX_LABEL_HORIZON_MS = MAX_LABEL_HORIZON_H * _H_MS

# Period boundaries, UTC, [start, end) in epoch milliseconds.
TRAIN_START_MS = 0                    # open on the left: take whatever data starts
HOLDOUT1_START_MS = 1_735_689_600_000  # 2025-01-01 00:00:00 UTC
HOLDOUT2_START_MS = 1_767_225_600_000  # 2026-01-01 00:00:00 UTC

_OPEN_END = 1 << 61                    # sentinel: "runs to the end of the data"

# ---------------------------------------------------------------- the wall --
# Opening the wall is how a holdout gets EVALUATED, and it has to be possible
# exactly once, on purpose, and never by accident. So it is gated on an
# environment variable rather than a function argument: nothing in the research
# path can set it by mistake, and every process that has it set says so out
# loud on import.
#
# MFT_OPEN_WALL=holdout1 moves the wall to 2026-01-01, which makes 2025
# buildable and evaluable while leaving 2026 sealed. There is deliberately no
# option that opens both at once.
#
# The override moves THE WALL. It must never move the split BOUNDARIES, which
# are facts about the calendar: holdout 1 is 2025 whether or not it is being
# evaluated. Keeping the true dates under separate names is not tidiness --
# without them, opening the wall would make `HOLDOUT1_START_MS` read 2026, and
# anything asking "where does holdout 1 begin?" would silently get an empty
# holdout and evaluate nothing.
TRUE_HOLDOUT1_MS = HOLDOUT1_START_MS
TRUE_HOLDOUT2_MS = HOLDOUT2_START_MS

_OPEN_WALL = os.environ.get("MFT_OPEN_WALL", "").strip().lower()
if _OPEN_WALL == "holdout1":
    HOLDOUT1_START_MS = HOLDOUT2_START_MS
    print("*** MFT_OPEN_WALL=holdout1 -- the wall is at 2026-01-01. "
          "HOLDOUT 1 IS BEING SPENT. ***", file=sys.stderr)
elif _OPEN_WALL == "holdout2":
    # The last wall. Nothing is sealed after this, so there is no third
    # chance: whatever is measured under it is the final out-of-sample
    # evidence this project will ever have.
    #
    # A FINITE date, not the _OPEN_END sentinel. The build scripts use the wall
    # as an `arange` endpoint, and handing them 2^61 asks numpy for a 597 GiB
    # array. 2027-01-01 is past every byte of data on disk, so it is unbounded
    # in effect while remaining a number the builders can work with.
    _WALL_OPEN_MS = 1_798_761_600_000   # 2027-01-01 00:00:00 UTC
    HOLDOUT1_START_MS = _WALL_OPEN_MS
    HOLDOUT2_START_MS = _WALL_OPEN_MS
    print("*** MFT_OPEN_WALL=holdout2 -- ALL WALLS OPEN. HOLDOUT 2 IS BEING "
          "SPENT. There is no further out-of-sample data. ***", file=sys.stderr)
elif _OPEN_WALL:
    raise ValueError(f"MFT_OPEN_WALL={_OPEN_WALL!r} is not recognised; "
                     f"accepted values are 'holdout1' and 'holdout2'")
HOLDOUT2_END_MS = _OPEN_END            # open on the right

# Which holdouts have been SPENT, in order. This is bookkeeping, and it is the
# only honest way to run a seal check: the seal must protect what is still
# unspent, not a date that was correct once. Leaving it fixed at 2025 made the
# check fail forever after holdout 1 was evaluated, and a check that always
# fails stops being read -- which is worse than not having one.
#
# Append to this list ONLY when a holdout has actually been evaluated.
SPENT_HOLDOUTS = ("holdout1", "holdout2")   # both spent; nothing is sealed


def seal_ms() -> int:
    """First boundary that must still be protected. Returns _OPEN_END when every
    holdout has been spent, i.e. there is nothing left to seal."""
    if "holdout1" not in SPENT_HOLDOUTS:
        return TRUE_HOLDOUT1_MS
    if "holdout2" not in SPENT_HOLDOUTS:
        return TRUE_HOLDOUT2_MS
    return _OPEN_END


TRAIN = "train"
HOLDOUT1 = "holdout1"
HOLDOUT2 = "holdout2"
SPLIT_NAMES = (TRAIN, HOLDOUT1, HOLDOUT2)


@dataclass(frozen=True)
class Split:
    """One period. `start_ms` inclusive, `end_ms` exclusive (nominal bounds)."""

    name: str
    start_ms: int
    end_ms: int

    @property
    def is_open_ended(self) -> bool:
        """True for the trailing split, whose end is the data end, not a date."""
        return self.end_ms >= _OPEN_END

    def usable_end_ms(
        self,
        horizon_hours: int = MAX_LABEL_HORIZON_H,
        data_end_ms: int | None = None,
    ) -> int:
        """Last bar_close_time whose label still resolves inside this period.

        For the open-ended split the binding constraint is the end of the DATA,
        not a calendar boundary: its final `horizon` hours of bars have labels
        that have not happened yet. Pass `data_end_ms` to purge them.
        """
        end = self.end_ms
        if data_end_ms is not None:
            end = min(end, int(data_end_ms))
        return end - horizon_hours * _H_MS

    def label(self) -> str:
        lo = _fmt(self.start_ms) if self.start_ms else "data start"
        return f"{self.name}: [{lo}, {_fmt(self.end_ms)})"


SPLITS: dict[str, Split] = {
    TRAIN: Split(TRAIN, TRAIN_START_MS, HOLDOUT1_START_MS),
    HOLDOUT1: Split(HOLDOUT1, HOLDOUT1_START_MS, HOLDOUT2_START_MS),
    HOLDOUT2: Split(HOLDOUT2, HOLDOUT2_START_MS, HOLDOUT2_END_MS),
}

# Back-compat alias for the one surviving caller (scripts/build_panel_features.py),
# which truncates its input to the research period by default.
RESEARCH_END_MS = SPLITS[TRAIN].usable_end_ms()


def _fmt(ms: int) -> str:
    if ms >= _OPEN_END:
        return "data end"
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")


def get(name: str) -> Split:
    if name not in SPLITS:
        raise KeyError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    return SPLITS[name]


def assign(close_time_ms: "pd.Series") -> "pd.Series":
    """Map each bar_close_time to its split name (ignores purging)."""
    out = pd.Series(TRAIN, index=close_time_ms.index, dtype=object)
    out[close_time_ms >= HOLDOUT1_START_MS] = HOLDOUT1
    out[close_time_ms >= HOLDOUT2_START_MS] = HOLDOUT2
    return out


def mask(
    close_time_ms: "pd.Series",
    name: str,
    horizon_hours: int = MAX_LABEL_HORIZON_H,
    purge: bool = True,
    data_end_ms: int | None = None,
) -> "pd.Series":
    """Boolean mask selecting rows of `name`, purged by `horizon_hours`.

    With purge=False the mask is the raw period, which is appropriate only for
    inspection -- never for fitting or for scoring a label-bearing row.

    For the open-ended split, `data_end_ms` defaults to the last bar present in
    `close_time_ms`, so its trailing rows -- whose labels have not happened yet
    -- are purged like any other boundary.
    """
    sp = get(name)
    if not purge:
        hi = sp.end_ms
    else:
        if data_end_ms is None and sp.is_open_ended and len(close_time_ms):
            data_end_ms = int(close_time_ms.max())
        hi = sp.usable_end_ms(horizon_hours, data_end_ms)
    return (close_time_ms >= sp.start_ms) & (close_time_ms < hi)


def take(
    df: "pd.DataFrame",
    name: str,
    horizon_hours: int = MAX_LABEL_HORIZON_H,
    purge: bool = True,
    time_col: str = "bar_close_time",
    data_end_ms: int | None = None,
) -> "pd.DataFrame":
    """Return the rows of `df` belonging to split `name`."""
    if time_col not in df.columns:
        raise KeyError(
            f"{time_col!r} not in frame; pass time_col= (columns: {list(df.columns)[:8]}...)"
        )
    return df.loc[mask(df[time_col], name, horizon_hours, purge, data_end_ms)]


def assert_train_only(df: "pd.DataFrame", time_col: str = "bar_close_time") -> None:
    """Raise if `df` contains any row at or beyond the holdout wall.

    Call this at the top of anything that fits, tunes, selects features, or
    otherwise makes a decision. It is the cheap guard that the previous
    research layer did not have.
    """
    if time_col not in df.columns:
        raise KeyError(f"{time_col!r} not in frame; cannot verify the holdout wall")
    bad = df[time_col] >= HOLDOUT1_START_MS
    n = int(bad.sum())
    if n:
        first = _fmt(int(df.loc[bad, time_col].min()))
        raise HoldoutLeak(
            f"{n} row(s) at or past the holdout wall "
            f"({_fmt(HOLDOUT1_START_MS)}); earliest offender {first}. "
            f"Research must run on train only -- use splits.take(df, 'train')."
        )


class HoldoutLeak(AssertionError):
    """Raised when research-side data crosses the holdout wall."""
