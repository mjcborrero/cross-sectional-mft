"""Read access to the normalized parquet tree produced by the backfill.

One loader for every downstream module: concatenates a dataset's monthly/daily
parquet files for a symbol, sorted and deduplicated on the dataset's primary
timestamp column.
"""

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .specs import DATASETS


class ParquetStore:
    def __init__(self, normalized_dir: Path):
        self.normalized_dir = Path(normalized_dir)

    def contract_symbols(self, dataset: str, prefix: str) -> List[str]:
        """Dated-contract symbol folders under a dataset matching '<prefix>_NNNNNN'
        (e.g. delivery futures BTCUSDT_210326). Sorted by name (= by expiry)."""
        base = self.normalized_dir / dataset
        if not base.exists():
            return []
        out = [p.name for p in base.iterdir()
               if p.is_dir() and re.fullmatch(rf"{re.escape(prefix)}_\d{{6}}", p.name)]
        return sorted(out)

    def files(self, dataset: str, symbol: str) -> List[Path]:
        folder = self.normalized_dir / dataset / symbol
        if not folder.exists():
            return []
        return sorted(folder.glob("*.parquet"))

    def last_timestamp(self, dataset: str, symbol: str):
        """Newest stored primary timestamp (ms) across dump + tail files, or
        None when the symbol has no data. Reads only the last two files by
        name (chronological by naming convention; the '-tail' REST file sorts
        after every period file and is deleted once dumps cover it)."""
        spec = DATASETS[dataset]
        ts_col = spec.timestamp_columns[0]
        files = self.files(dataset, symbol)
        best = None
        for path in files[-2:]:
            col = pd.read_parquet(path, columns=[ts_col])[ts_col]
            if len(col):
                m = int(col.max())
                best = m if best is None or m > best else best
        return best

    def load(self, dataset: str, symbol: str,
             columns: Optional[List[str]] = None,
             start: Optional[str] = None,
             end: Optional[str] = None) -> pd.DataFrame:
        """Load one dataset/symbol as a single sorted, deduplicated frame.

        start/end are inclusive 'YYYY-MM' bounds applied at file level (cheap
        pre-filter on the filename period; daily files match by their month).
        """
        spec = DATASETS[dataset]
        ts_col = spec.timestamp_columns[0]
        # The sort/dedup below is a hard guarantee, not best-effort: monthly
        # and daily archives can overlap on disk (a monthly supersedes dailies
        # in the PLANNER, but already-converted dailies stay), so a projected
        # read that omitted ts_col used to skip dedup and return duplicated,
        # unsorted rows -- silently corrupting every searchsorted/as-of
        # consumer (found 2026-07-21 by the live-panel parity gate). Always
        # load ts_col; drop it again if the caller didn't ask for it.
        drop_ts = columns is not None and ts_col not in columns
        if drop_ts:
            columns = [ts_col] + list(columns)
        frames = []
        for path in self.files(dataset, symbol):
            match = re.search(r"(\d{4}-\d{2})(?:-\d{2})?$", path.stem)
            month = match.group(1) if match else ""
            # Files without a period token (the REST '-tail' file) are always
            # loaded: they hold the most recent rows by construction.
            if start and month and month < start:
                continue
            if end and month and month > end:
                continue
            frames.append(pd.read_parquet(path, columns=columns))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if ts_col in df.columns:
            df = (df.sort_values(ts_col, kind="stable")
                    .drop_duplicates(subset=ts_col, keep="last")
                    .reset_index(drop=True))
        if drop_ts:
            df = df.drop(columns=[ts_col])
        return df
