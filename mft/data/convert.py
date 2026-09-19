"""Convert a downloaded Binance zip (one CSV inside) to a normalized parquet file.

Normalization absorbs the known schema drift in the archives:
- Header row present in newer files, absent in older ones -> detected per file.
- Header column names differ from internal names (e.g. 'count' vs 'trade_count')
  -> alias map.
- Spot timestamps switched from milliseconds to microseconds in January 2025
  -> detected by magnitude, normalized to UTC milliseconds.
"""

import zipfile
from pathlib import Path

import pandas as pd

from .specs import DatasetSpec

# Header names as published by Binance -> internal canonical names.
_HEADER_ALIASES = {
    "count": "trade_count",
    "taker_buy_volume": "taker_buy_base_volume",
}

# Epoch values above this are microseconds (ms epochs are ~1.8e12 in the 2020s,
# microsecond epochs ~1.8e15).
_MICROSECOND_THRESHOLD = 10 ** 14


def _has_header(zip_path: Path) -> bool:
    """A data row starts with a number; a header row does not."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            first_field = fh.readline().decode("utf-8", "replace").split(",")[0].strip()
    try:
        float(first_field)
        return False
    except ValueError:
        return True


def convert_zip(zip_path: Path, spec: DatasetSpec, out_path: Path) -> int:
    """Read the zipped CSV, normalize, write parquet. Returns the row count."""
    header = 0 if _has_header(zip_path) else None
    df = pd.read_csv(zip_path, header=header)

    if header is None:
        names = list(spec.columns)[: df.shape[1]]
        names += [f"extra_{i}" for i in range(df.shape[1] - len(names))]
        df.columns = names
    else:
        cols = [str(c).strip().lower() for c in df.columns]
        df.columns = [_HEADER_ALIASES.get(c, c) for c in cols]

    df = df.drop(columns=["ignore"], errors="ignore")

    ts_cols = [c for c in spec.timestamp_columns if c in df.columns]
    for col in ts_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().all():
            # Some datasets (metrics) publish datetime strings, not epochs.
            # as_unit("ms") pins the resolution explicitly: pandas 3 parses to
            # microseconds, so int64 // 1e6 would silently yield seconds.
            parsed = pd.to_datetime(df[col], utc=True, format="mixed")
            ts = parsed.dt.as_unit("ms").astype("int64")
        else:
            ts = numeric.astype("int64")
            if len(ts) and abs(ts.iloc[0]) > _MICROSECOND_THRESHOLD:
                ts = ts // 1000
        df[col] = ts

    if ts_cols:
        df = df.sort_values(ts_cols[0], kind="stable").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd")
    return len(df)
