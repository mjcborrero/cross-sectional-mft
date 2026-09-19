"""Compute what to download: wanted (per listing API) minus have (per manifest).

Policy encoded here: prefer monthly archives for deep history; use daily files
only for months newer than the latest available monthly archive (the current
month is never in the monthly archives yet).
"""

import re
from dataclasses import dataclass
from typing import List, Set

from .client import BinanceVisionClient
from .specs import DatasetSpec

# Trailing date token in archive filenames: YYYY-MM (monthly) or YYYY-MM-DD (daily).
_PERIOD_RE = re.compile(r"-(\d{4}-\d{2}(?:-\d{2})?)\.zip$")


@dataclass(frozen=True)
class PlanItem:
    dataset: str
    symbol: str
    period: str       # YYYY-MM or YYYY-MM-DD
    granularity: str  # 'monthly' | 'daily'
    key: str          # full object key


def _period_of(key: str) -> str:
    match = _PERIOD_RE.search(key)
    return match.group(1) if match else ""


def plan_symbol(client: BinanceVisionClient, spec: DatasetSpec, symbol: str,
                start: str, end: str, done_keys: Set[str]) -> List[PlanItem]:
    """Plan one (dataset, symbol). start/end are inclusive YYYY-MM bounds."""
    items: List[PlanItem] = []
    last_monthly = ""  # latest month covered by a monthly archive, regardless of manifest

    if "monthly" in spec.granularities:
        for key in client.list_keys(spec.prefix("monthly", symbol)):
            if not key.endswith(".zip"):
                continue
            period = _period_of(key)
            if not period or not (start <= period <= end):
                continue
            last_monthly = max(last_monthly, period)
            if key not in done_keys:
                items.append(PlanItem(spec.name, symbol, period, "monthly", key))

    if "daily" in spec.granularities:
        for key in client.list_keys(spec.prefix("daily", symbol)):
            if not key.endswith(".zip"):
                continue
            period = _period_of(key)
            month = period[:7]
            if not period or not (start <= month <= end):
                continue
            # Daily files only fill the tail beyond monthly coverage. When the
            # dataset has no monthly granularity at all, take every daily file.
            if "monthly" in spec.granularities and month <= last_monthly:
                continue
            if key not in done_keys:
                items.append(PlanItem(spec.name, symbol, period, "daily", key))

    return items


def discover_delivery_symbols(client: BinanceVisionClient, spec: DatasetSpec,
                              underlying: str) -> List[str]:
    """Enumerate dated contracts (e.g. BTCUSDT_250626) for an underlying.

    Listing the dataset's symbol directory and filtering beats guessing expiry
    dates: contracts appear and disappear quarterly.
    """
    pattern = re.compile(rf"^{re.escape(underlying)}_\d{{6}}$")
    granularity = spec.granularities[0]
    # Symbol directories live one level above the interval segment.
    base = "/".join(["data", spec.market, granularity, spec.data_type]) + "/"
    symbols = []
    for prefix in client.list_prefixes(base):
        name = prefix.rstrip("/").rsplit("/", 1)[-1]
        if pattern.match(name):
            symbols.append(name)
    return sorted(symbols)
