"""Orchestrates the backfill: download -> verify -> convert -> purge -> record.

Disk policy: raw zips are deleted immediately after successful conversion (the
manifest keeps the SHA256), so peak raw usage is one file per worker.
"""

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .client import BinanceVisionClient
from .convert import convert_zip
from .manifest import Manifest
from .planner import PlanItem
from .specs import DATASETS


@dataclass
class BackfillStats:
    ok: int = 0
    failed: int = 0
    rows: int = 0
    errors: List[str] = field(default_factory=list)


class BackfillRunner:
    def __init__(self, client: BinanceVisionClient, manifest: Manifest,
                 raw_dir: Path, normalized_dir: Path, workers: int = 4):
        self.client = client
        self.manifest = manifest
        self.raw_dir = raw_dir
        self.normalized_dir = normalized_dir
        self.workers = workers

    def _process(self, item: PlanItem) -> int:
        spec = DATASETS[item.dataset]
        zip_path = self.raw_dir / Path(item.key).name
        out_path = (self.normalized_dir / item.dataset / item.symbol
                    / f"{item.symbol}-{item.period}.parquet")
        try:
            self.client.download(item.key, zip_path)
            sha = self.client.verify(item.key, zip_path)
            rows = convert_zip(zip_path, spec, out_path)
        finally:
            zip_path.unlink(missing_ok=True)
        self.manifest.record(
            key=item.key, dataset=item.dataset, symbol=item.symbol,
            period=item.period, granularity=item.granularity,
            status="converted", sha256=sha, rows=rows, parquet_path=str(out_path),
        )
        return rows

    def run(self, plan: List[PlanItem]) -> BackfillStats:
        stats = BackfillStats()
        total = len(plan)
        if total == 0:
            return stats
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._process, item): item for item in plan}
            for done, future in enumerate(as_completed(futures), start=1):
                item = futures[future]
                try:
                    stats.rows += future.result()
                    stats.ok += 1
                except Exception as exc:  # record and continue; never abort the batch
                    stats.failed += 1
                    msg = f"{item.key}: {exc}"
                    stats.errors.append(msg)
                    self.manifest.record(
                        key=item.key, dataset=item.dataset, symbol=item.symbol,
                        period=item.period, granularity=item.granularity,
                        status="failed", error=f"{exc}\n{traceback.format_exc()}",
                    )
                if done % 25 == 0 or done == total:
                    print(f"  [{done}/{total}] ok={stats.ok} failed={stats.failed}")
        return stats
