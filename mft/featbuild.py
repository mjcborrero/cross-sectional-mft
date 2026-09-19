"""Feature-building harness for the 8h decision grid.

Every feature family is built through this, so the guarantees are structural
rather than remembered. Post-mortem rules made mechanical:

  GATES ABORT, THEY DO NOT PRINT.  A family that fails any Stage 0 check
  raises; nothing is written. A diagnostic that is printed and not read is not
  a diagnostic.

  INVARIANTS ARE ASSERTIONS IN THE INNER LOOP.  Point-in-time is enforced by
  the loader -- a family physically cannot see data past its cutoff, because
  the context filters on load rather than trusting the family to filter.

  THE TRUNCATION TEST.  Build twice, once on all data and once on data
  truncated at a midpoint, and require the overlapping rows to be IDENTICAL.
  Deleting future data cannot change a genuinely point-in-time value. This
  catches what timestamp assertions cannot: full-sample normalisation, any
  parameter fitted on all data, a rolling window that includes the current
  row. Gate 8B was the same idea applied to the target and it caught a 94%
  error.

Bar-level (context) features are declared, not inferred: a family names them
in `context_columns` and they are exempted from the within-bar variation
check instead of being silently deleted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from mft import splits
from mft.data.store import ParquetStore
from mft.paths import DATA_DIR, NORMALIZED_DIR

MS_MIN = 60_000
MS_HOUR = 3_600_000
H = 8

# Stage 0 thresholds -- pre-registered in docs/FEATURE_EVALUATION.md
MIN_COVERAGE = 0.95
MIN_COIN_COVERAGE = 0.80
MIN_WITHIN_BAR_VAR_FRAC = 0.95
MIN_DISTINCT_PER_BAR = 10


class GateFailure(AssertionError):
    """A Stage 0 gate failed. Nothing is written."""


@dataclass
class GridContext:
    """Point-in-time data access for one build.

    `cutoff_ms` is the only thing that makes the truncation test possible: with
    it set, EVERY loader silently drops rows at or after the cutoff, so a
    family cannot accidentally see beyond it even if it tries.
    """

    t_obs: np.ndarray
    cutoff_ms: int | None = None
    _store: ParquetStore = field(default_factory=lambda: ParquetStore(NORMALIZED_DIR))

    @property
    def wall(self) -> int:
        return splits.HOLDOUT1_START_MS

    def _clip(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        df = df[df[col] < self.wall]
        if self.cutoff_ms is not None:
            df = df[df[col] <= self.cutoff_ms]
        return df

    def instants(self) -> np.ndarray:
        t = self.t_obs
        return t if self.cutoff_ms is None else t[t <= self.cutoff_ms]

    def dataset(self, name: str, symbol: str, columns: list[str],
                ts_col: str) -> pd.DataFrame:
        df = self._store.load(name, symbol, columns=columns)
        return self._clip(df, ts_col).sort_values(ts_col)

    def grid(self) -> pd.DataFrame:
        g = pd.read_parquet(DATA_DIR / "grid/decision_grid_8h.parquet")
        return self._clip(g, "t_obs")

    def dataset_start(self, name: str, symbol: str) -> int | None:
        """First month a dataset exists for a symbol, from FILENAMES.

        Cheap -- no parquet is read. Used to build an honest coverage
        denominator: a feature cannot be computed before its input exists, and
        that absence is a property of the DATA, not of the feature, so it
        cannot be gamed by a broken computation.
        """
        files = self._store.files(name, symbol)
        if not files:
            return None
        # Anchor on the PERIOD suffix. A delivery-contract filename embeds the
        # expiry too -- "BTCUSDT_210326-2020-12.parquet" -- and an unanchored
        # match takes "0326-20" from the contract code and produces month 20.
        stem = files[0].name.split(".")[0]
        if stem.endswith("-tail"):
            stem = stem[: -len("-tail")]
        # Three filename shapes exist and the period is ALWAYS the suffix:
        #   monthly            AAVEUSDT-2020-10
        #   daily              AAVEUSDT-2021-12-01      (metrics)
        #   contract + monthly BTCUSDT_210326-2021-02   (delivery futures)
        # Anchoring matters: an unanchored match takes "0326-20" out of the
        # delivery contract's expiry code and yields month 20.
        m = re.search(r"(\d{4})-(\d{2})(?:-(\d{2}))?$", stem)
        if not m or not 1 <= int(m.group(2)) <= 12:
            return None
        day = m.group(3) or "01"
        return int(pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{day}",
                                tz="UTC").value // 10**6)

    def artifact(self, name: str, ts_col: str = "t_obs") -> pd.DataFrame:
        """A prebuilt grid artifact, clipped to the cutoff.

        Safe under the truncation test only for artifacts whose rows are
        themselves point-in-time -- e.g. returns_1h, where the value at hour t
        depends on prices at t-1 and t and nothing later. Clipping such a file
        is equivalent to having rebuilt it truncated.
        """
        g = pd.read_parquet(DATA_DIR / f"grid/{name}.parquet")
        return self._clip(g, ts_col)

    def target_side(self, name: str) -> pd.DataFrame:
        """Beta/idio-vol or index artifacts. Never the target itself."""
        assert "target" not in name, "feature builds must not read the target"
        g = pd.read_parquet(DATA_DIR / f"grid/{name}.parquet")
        return self._clip(g, "t_obs")


class Family:
    """One feature family. Subclasses implement `compute`."""

    code: str = "?"
    name: str = "unnamed"
    context_columns: tuple[str, ...] = ()      # bar-level, exempt from Stage 1
    sector_columns: tuple[str, ...] = ()       # per-sector tier, coarse but real
    change_columns: tuple[str, ...] = ()       # transient by construction (Stage 3)
    input_datasets: tuple[str, ...] = ()       # sets the coverage denominator
    warmup_bars: int = 0                       # longest trailing window, in grid steps

    def compute(self, ctx: GridContext) -> pd.DataFrame:
        """Return a frame indexed by (t_obs, symbol) with one column per feature."""
        raise NotImplementedError

    def feature_availability(self, ctx: GridContext) -> dict[str, int | dict[str, int]]:
        """Per-FEATURE input start, for features whose source begins later than
        the family's.

        Returned values must be DERIVED FROM THE DATA (a dataset's first file),
        never hardcoded, so a feature cannot nominate a convenient denominator
        for itself. Features not listed use the family-wide denominator.

        A value may be an int (one start for every symbol) or a symbol -> start
        mapping. Per-symbol is the honest form whenever the source dataset
        begins at different times for different coins: `metrics` starts 2020-09
        for BTCUSDT, 2021-12 for eighteen others and 2023-05 for SUIUSDT, and
        collapsing that to one scalar would either discard eighteen months of
        real data or excuse a coin that has none.
        """
        return {}


# ----------------------------------------------------------------- gates ----

def _gate(ok: bool, label: str, detail: str, failures: list[str]) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def stage0(df: pd.DataFrame, fam: Family, avail: dict[str, int] | None = None) -> None:
    """Stage 0 + 1 of docs/FEATURE_EVALUATION.md. Raises on failure.

    `avail` maps symbol -> first instant its INPUT data exists. Coverage is
    measured inside that window only. A feature cannot be computed before its
    source dataset is published, and that is a fact about the data, not a
    defect in the feature -- the gate exists to catch broken computations, and
    conflating the two would fail sound features while passing broken ones.
    """
    failures: list[str] = []
    feats = [c for c in df.columns if c not in ("t_obs", "symbol")]
    ctxcols = set(fam.context_columns)

    if fam.warmup_bars:
        # A rolling estimator cannot produce a value before its window fills.
        # That is a declared property of the construction, visible in code, not
        # something a broken feature can claim for itself -- and it is applied
        # uniformly, so a short-window feature gains nothing by it.
        starts = df.groupby("symbol")["t_obs"].transform("min")
        cutoff = starts + fam.warmup_bars * 8 * MS_HOUR
        warm = df["t_obs"] < cutoff
        print(f"\n  declared warm-up: {fam.warmup_bars} bars; "
              f"{int(warm.sum()):,} of {len(df):,} rows excluded from the "
              f"coverage denominator")
        df = df[~warm]

    if avail:
        lo = df["symbol"].map(avail)
        eligible = lo.isna() | (df["t_obs"] >= lo)
        dropped = int((~eligible).sum())
        first = min(avail.values())
        print(f"\n  input availability: from {splits._fmt(first)}; "
              f"{dropped:,} of {len(df):,} rows precede their inputs and are "
              f"excluded from the coverage denominator")
        df = df[eligible]

    print(f"\nSTAGE 0 -- validity  ({len(feats)} features, {len(df):,} rows)")
    print("=" * 74)

    per_feat = fam.feature_availability(GridContext(np.array([], dtype=np.int64)))         if hasattr(fam, "feature_availability") else {}
    if per_feat:
        for k, v in per_feat.items():
            lo = min(v.values()) if isinstance(v, dict) else v
            n = f" (per-symbol, {len(v)} coins)" if isinstance(v, dict) else ""
            print(f"  [note] {k}: inputs begin {splits._fmt(lo)}{n}; coverage "
                  f"measured from there")

    for c in feats:
        # The pooled and per-coin gates MUST share one denominator. Applying the
        # availability window to the pooled gate only would let a feature pass
        # on aggregate and then fail per-coin for a reason the harness has
        # already accepted as a data fact rather than a defect.
        sub_df = df
        if c in per_feat:
            v = per_feat[c]
            lo = df["symbol"].map(v) if isinstance(v, dict) else v
            sub_df = df[df["t_obs"] >= lo] if not isinstance(v, dict) else \
                df[pd.notna(lo) & (df["t_obs"] >= lo.fillna(np.inf))]
        col = sub_df[c]
        cov = col.notna().mean()
        _gate(cov >= MIN_COVERAGE, f"{c}: coverage >= {MIN_COVERAGE:.0%}",
              f"{cov:.1%}", failures)
        _gate(bool(np.isfinite(col.dropna()).all()), f"{c}: finite", "", failures)
        _gate(float(col.std(skipna=True)) > 0, f"{c}: not constant", "", failures)

        per_coin = sub_df.groupby("symbol")[c].apply(lambda s: s.notna().mean())
        worst = float(per_coin.min())
        _gate(worst >= MIN_COIN_COVERAGE,
              f"{c}: per-coin coverage >= {MIN_COIN_COVERAGE:.0%}",
              f"worst {per_coin.idxmin()} {worst:.1%}", failures)

    print(f"\nSTAGE 1 -- ranking power")
    print("=" * 74)
    for c in feats:
        if c in ctxcols:
            print(f"  [ctx ] {c}: declared bar-level, exempt")
            continue
        # Only bars where the feature EXISTS can be judged on ranking power.
        # A bar in which every coin is NaN has nothing to rank and is not
        # evidence of a constant feature.
        sub = df[df[c].notna()]
        g = sub.groupby("t_obs")[c]
        sizes = g.size()
        rankable = sizes[sizes >= 2].index
        g = sub[sub["t_obs"].isin(rankable)].groupby("t_obs")[c]
        if len(rankable) == 0:
            _gate(False, f"{c}: has any rankable bar", "none", failures)
            continue
        var_frac = float((g.std() > 0).mean())
        _gate(var_frac >= MIN_WITHIN_BAR_VAR_FRAC,
              f"{c}: within-bar variation in >= {MIN_WITHIN_BAR_VAR_FRAC:.0%} of bars",
              f"{var_frac:.1%}", failures)
        distinct = float(g.nunique().median())
        # Three tiers of ranking power (FEATURE_EXPLORATION.md §12). A
        # per-SECTOR feature can never take more values than there are sectors
        # -- coarse, but genuine within-bar discrimination, and unavailable to
        # any market-wide aggregate. Holding it to the per-coin threshold would
        # delete a tier the design deliberately keeps. Declared by the family,
        # not inferred, so it cannot be claimed after the fact.
        is_sector = c in set(fam.sector_columns)
        need = 4 if is_sector else MIN_DISTINCT_PER_BAR
        _gate(distinct >= need,
              f"{c}: >= {need} distinct values per bar"
              + ("  [sector tier]" if is_sector else ""),
              f"median {distinct:.0f}", failures)

    if failures:
        raise GateFailure(f"{len(failures)} gate(s) failed: " + "; ".join(failures[:6]))


def truncation_test(fam: Family, t_obs: np.ndarray, frac: float = 0.6) -> None:
    """Build twice and require the overlap to be bit-identical. Raises on failure.

    A genuinely point-in-time value cannot change when future data is deleted.
    """
    # The cut must fall AFTER every coin has listed, or the truncated build has
    # a narrower cross-section than the full one and any cross-sectionally
    # normalised feature differs in the last bit -- pandas sums a 20-column
    # frame (one of them all-NaN) differently from a 19-column frame. Measured:
    # median 6.6e-16, one ULP, amplified to 1e-13 through a variance ratio.
    #
    # That is not a leak, but tolerating it would mean tolerating 1e-13
    # everywhere, and a real bounded lookahead is only ~1e-2. Removing the
    # cause keeps the gate at exact equality, which is the useful setting.
    # A later cut also compares MORE shared rows, so the test gets stronger.
    _g = pd.read_parquet(DATA_DIR / "grid/decision_grid_8h.parquet",
                         columns=["symbol", "t_obs"])
    _g = _g[_g["t_obs"] < splits.HOLDOUT1_START_MS]
    last_listing = int(_g.groupby("symbol")["t_obs"].min().max())
    cut = int(max(np.quantile(t_obs, frac), last_listing + 30 * 24 * MS_HOUR))
    full = fam.compute(GridContext(t_obs)).sort_index()
    trunc = fam.compute(GridContext(t_obs, cutoff_ms=cut)).sort_index()

    common = full.index.intersection(trunc.index)
    a, b = full.loc[common], trunc.loc[common]

    print(f"\nTRUNCATION TEST  (cut at {splits._fmt(cut)}, {len(common):,} shared rows)")
    print("=" * 74)
    failures: list[str] = []
    for c in a.columns:
        x, y = a[c], b[c]
        both = x.notna() & y.notna()
        same_na = bool((x.isna() == y.isna()).all())
        maxdiff = float((x[both] - y[both]).abs().max()) if both.any() else 0.0
        ok = same_na and (maxdiff == 0.0)
        _gate(ok, f"{c}: identical under truncation",
              f"max|diff| {maxdiff:.2e}" + ("" if same_na else ", NaN pattern differs"),
              failures)
    if failures:
        raise GateFailure(
            "TRUNCATION TEST FAILED -- the feature sees data past its stamp: "
            + "; ".join(failures[:4]))


def run(fam: Family, out_name: str, skip_truncation: bool = False) -> pd.DataFrame:
    """Build one family, gate it, and write it. Nothing is written on failure."""
    grid = pd.read_parquet(DATA_DIR / "grid/decision_grid_8h.parquet")
    t_obs = np.sort(grid["t_obs"].unique())

    print(f"Family {fam.code} -- {fam.name}")
    print(f"  {len(t_obs):,} decision instants")

    df = fam.compute(GridContext(t_obs)).sort_index()

    # Restrict to (t_obs, symbol) pairs that EXIST in the decision grid. A coin
    # that had not listed yet is not a missing value -- it is not an
    # opportunity. Measuring coverage over the cartesian product would make the
    # per-coin floor unreachable for any coin listed mid-sample (SUIUSDT is
    # present in only 39% of instants), which is a denominator error, not a
    # data problem.
    universe = pd.MultiIndex.from_frame(grid[["t_obs", "symbol"]])
    df = df.reindex(df.index.intersection(universe))
    flat = df.reset_index()

    splits.assert_train_only(flat.assign(bar_close_time=flat["t_obs"]))

    avail = None
    if fam.input_datasets:
        ctx0 = GridContext(t_obs)
        avail = {}
        for sym in sorted(grid["symbol"].unique()):
            starts = [ctx0.dataset_start(d, sym) for d in fam.input_datasets]
            starts = [x for x in starts if x is not None]
            if starts:
                avail[sym] = max(starts)
    stage0(flat, fam, avail)
    if not skip_truncation:
        truncation_test(fam, t_obs)

    path = DATA_DIR / f"features/{out_name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    flat.to_parquet(path, index=False)
    print(f"\nALL GATES PASSED. Wrote {path}  ({path.stat().st_size/1e6:.2f} MB)")
    return flat
