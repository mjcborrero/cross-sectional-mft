"""Lock the train/holdout boundaries and the purge rule.

These are cheap tests guarding an expensive mistake: a boundary that quietly
moves, or a purge that quietly stops applying, reintroduces exactly the class
of contamination this project already paid for once.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mft import splits

H_MS = 3_600_000


def ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").value // 1_000_000)


# --------------------------------------------------------------------- bounds

def test_boundaries_are_the_intended_calendar_dates():
    assert splits.HOLDOUT1_START_MS == ms("2025-01-01")
    assert splits.HOLDOUT2_START_MS == ms("2026-01-01")
    assert splits.get("train").end_ms == ms("2025-01-01")
    assert splits.get("holdout1").start_ms == ms("2025-01-01")
    assert splits.get("holdout1").end_ms == ms("2026-01-01")
    assert splits.get("holdout2").start_ms == ms("2026-01-01")
    assert splits.get("holdout2").is_open_ended


def test_max_horizon_matches_the_label_ladder():
    # The longest label in the panel is fwd_return_120h; the triple barrier
    # caps at 48h. If a longer label is ever added this must be raised.
    assert splits.MAX_LABEL_HORIZON_H == 120


def test_train_usable_end_is_purged_by_the_full_horizon():
    assert splits.get("train").usable_end_ms() == ms("2024-12-27")
    assert splits.get("holdout1").usable_end_ms() == ms("2025-12-27")


def test_shorter_horizon_recovers_data():
    assert splits.get("train").usable_end_ms(24) == ms("2024-12-31")


# --------------------------------------------------------------------- assign

@pytest.mark.parametrize("stamp,expected", [
    ("2020-01-01", "train"),
    ("2024-12-31 23:59", "train"),
    ("2025-01-01 00:00", "holdout1"),
    ("2025-12-31 23:59", "holdout1"),
    ("2026-01-01 00:00", "holdout2"),
    ("2026-07-31", "holdout2"),
])
def test_assign_places_each_bar_in_the_right_period(stamp, expected):
    got = splits.assign(pd.Series([ms(stamp)]))
    assert got.iloc[0] == expected


def test_splits_partition_the_timeline():
    t = pd.Series([ms(d) for d in
                   ["2020-06-01", "2024-06-01", "2025-06-01", "2026-06-01"]])
    counts = splits.assign(t).value_counts()
    assert counts.sum() == len(t)
    masks = [splits.mask(t, n, purge=False) for n in splits.SPLIT_NAMES]
    # disjoint, and together they cover every row
    assert sum(int(m.sum()) for m in masks) == len(t)


# ---------------------------------------------------------------------- purge

def test_purge_excludes_bars_whose_label_crosses_the_boundary():
    end = splits.get("train").usable_end_ms()
    t = pd.Series([end - 1, end, end + 1])
    kept = splits.mask(t, "train")
    assert list(kept) == [True, False, False]


def test_purged_train_labels_resolve_before_holdout_starts():
    end = splits.get("train").usable_end_ms()
    latest_label_resolution = (end - 1) + splits.MAX_LABEL_HORIZON_H * H_MS
    assert latest_label_resolution < splits.HOLDOUT1_START_MS


def test_purge_false_keeps_the_whole_period():
    t = pd.Series([ms("2024-12-31 23:00")])
    assert not splits.mask(t, "train").iloc[0]
    assert splits.mask(t, "train", purge=False).iloc[0]


def test_open_ended_split_purges_against_the_data_end():
    data_end = ms("2026-07-31")
    t = pd.Series([ms("2026-07-20"), ms("2026-07-30"), data_end])
    kept = splits.mask(t, "holdout2", data_end_ms=data_end)
    # only the bar more than 120h before the data end survives
    assert list(kept) == [True, False, False]


def test_open_ended_purge_infers_data_end_from_the_series():
    t = pd.Series([ms("2026-07-20"), ms("2026-07-30"), ms("2026-07-31")])
    assert list(splits.mask(t, "holdout2")) == [True, False, False]


# ---------------------------------------------------------------------- guard

def test_assert_train_only_passes_on_train_rows():
    df = pd.DataFrame({"bar_close_time": [ms("2023-05-01"), ms("2024-01-01")]})
    splits.assert_train_only(df)


def test_assert_train_only_raises_on_a_holdout_row():
    df = pd.DataFrame({"bar_close_time": [ms("2024-01-01"), ms("2025-01-01")]})
    with pytest.raises(splits.HoldoutLeak) as e:
        splits.assert_train_only(df)
    assert "2025-01-01" in str(e.value)


def test_assert_train_only_raises_when_the_time_column_is_missing():
    with pytest.raises(KeyError):
        splits.assert_train_only(pd.DataFrame({"nope": [1]}))


# ----------------------------------------------------------------------- take

def test_take_selects_only_the_named_split():
    df = pd.DataFrame({"bar_close_time": [ms("2023-01-01"), ms("2025-06-01"),
                                          ms("2026-03-01")]})
    assert len(splits.take(df, "train")) == 1
    assert len(splits.take(df, "holdout1")) == 1
    assert len(splits.take(df, "holdout2", data_end_ms=ms("2026-12-01"))) == 1


def test_take_rejects_an_unknown_split_name():
    with pytest.raises(KeyError):
        splits.take(pd.DataFrame({"bar_close_time": [0]}), "test")


def test_take_reports_a_missing_time_column():
    with pytest.raises(KeyError):
        splits.take(pd.DataFrame({"t": [0]}), "train")
