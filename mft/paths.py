"""Central data location.

All market data and every derived artifact live OUTSIDE the repository, under
one directory. Nothing here is committed: source data is downloaded from
Binance's public archive by `scripts/backfill.py`, and everything downstream
(grid, index, betas, targets, features, results) is rebuilt from it.

Set the location with the MFT_DATA_DIR environment variable. The default is a
`data/` directory beside this package, which `.gitignore` excludes.

Layout produced by the pipeline:

  normalized/   source market data, one parquet per (dataset, symbol, day)
  manifest.db   ingestion bookkeeping, so backfill knows what is on disk
  grid/         decision grid, weights, index, betas, forward legs, targets
  features/     feature families, the frozen list, and every result JSON
  train/        train-only copy of the panel (see materialize_train.py)
"""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("MFT_DATA_DIR",
                               Path(__file__).resolve().parent.parent / "data"))

# Source data. Nothing here writes to it; backfill appends only.
NORMALIZED_DIR = DATA_DIR / "normalized"

# Train-only copy of the panel, written by scripts/materialize_train.py. Every
# file under it ends at the purged train boundary, so a study pointed here
# cannot read holdout1 or holdout2 even by accident. Research should default
# to this directory; only end-of-programme evaluation reads DATA_DIR directly.
TRAIN_DIR = DATA_DIR / "train"
