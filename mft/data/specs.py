"""Declarative catalog of the Binance public datasets used by the backfill pipeline.

Each DatasetSpec describes one dataset on data.binance.vision: where it lives,
which file granularities exist (monthly/daily), and how to interpret its CSV
contents (column names for headerless files, which columns are epoch timestamps).

Adding a new dataset means adding a spec here, not writing new code.

Known schema drift absorbed downstream (see convert.py):
- Some files carry a header row, older ones do not.
- Spot data switched from millisecond to microsecond timestamps in January 2025.
- Header column names differ slightly from internal names (alias map in convert.py).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

DOWNLOAD_BASE = "https://data.binance.vision/"
LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
)

FUNDING_COLUMNS = ("calc_time", "funding_interval_hours", "last_funding_rate")

LIQUIDATION_COLUMNS = (
    "time", "side", "order_type", "time_in_force",
    "original_quantity", "price", "average_price", "order_status",
    "last_fill_quantity", "accumulated_fill_quantity",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str                          # internal dataset id (also the storage folder name)
    market: str                        # "spot", "futures/um", "futures/cm"
    data_type: str                     # path segment on data.binance.vision
    interval: Optional[str]            # kline interval path segment, or None
    granularities: Tuple[str, ...]     # subset of ("monthly", "daily")
    columns: Tuple[str, ...]           # column names when the CSV has no header
    timestamp_columns: Tuple[str, ...]  # epoch columns normalized to UTC milliseconds

    def prefix(self, granularity: str, symbol: str) -> str:
        parts = ["data", self.market, granularity, self.data_type, symbol]
        if self.interval:
            parts.append(self.interval)
        return "/".join(parts) + "/"


DATASETS = {
    "spot_klines": DatasetSpec(
        name="spot_klines", market="spot", data_type="klines", interval="1m",
        granularities=("monthly", "daily"),
        columns=KLINE_COLUMNS, timestamp_columns=("open_time", "close_time"),
    ),
    "perp_klines": DatasetSpec(
        name="perp_klines", market="futures/um", data_type="klines", interval="1m",
        granularities=("monthly", "daily"),
        columns=KLINE_COLUMNS, timestamp_columns=("open_time", "close_time"),
    ),
    "premium_index": DatasetSpec(
        name="premium_index", market="futures/um", data_type="premiumIndexKlines",
        interval="1m", granularities=("monthly", "daily"),
        columns=KLINE_COLUMNS, timestamp_columns=("open_time", "close_time"),
    ),
    "funding": DatasetSpec(
        name="funding", market="futures/um", data_type="fundingRate", interval=None,
        granularities=("monthly",),  # fundingRate is published monthly-only
        columns=FUNDING_COLUMNS, timestamp_columns=("calc_time",),
    ),
    "liquidations": DatasetSpec(
        name="liquidations", market="futures/um", data_type="liquidationSnapshot",
        interval=None, granularities=("daily",),  # liquidationSnapshot is daily-only
        columns=LIQUIDATION_COLUMNS, timestamp_columns=("time",),
    ),
    # Open interest + positioning snapshots, 5m granularity, ~2020-09 onward.
    # create_time is a datetime string in the raw files (convert.py parses it).
    "metrics": DatasetSpec(
        name="metrics", market="futures/um", data_type="metrics", interval=None,
        granularities=("daily",),
        columns=("create_time", "symbol", "sum_open_interest",
                 "sum_open_interest_value", "count_toptrader_long_short_ratio",
                 "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                 "sum_taker_long_short_vol_ratio"),
        timestamp_columns=("create_time",),
    ),
    # Delivery (dated) futures share the klines layout; symbols look like
    # BTCUSDT_250626 (USDT-margined) / BTCUSD_250626 (coin-margined) and are
    # discovered via the listing API, not configured.
    "um_delivery_klines": DatasetSpec(
        name="um_delivery_klines", market="futures/um", data_type="klines",
        interval="1m", granularities=("monthly", "daily"),
        columns=KLINE_COLUMNS, timestamp_columns=("open_time", "close_time"),
    ),
    "cm_delivery_klines": DatasetSpec(
        name="cm_delivery_klines", market="futures/cm", data_type="klines",
        interval="1m", granularities=("monthly", "daily"),
        columns=KLINE_COLUMNS, timestamp_columns=("open_time", "close_time"),
    ),
    # Documented upgrade path, deliberately NOT in DEFAULT_DATASETS (v1 is klines-only).
    "spot_aggtrades": DatasetSpec(
        name="spot_aggtrades", market="spot", data_type="aggTrades", interval=None,
        granularities=("monthly", "daily"),
        columns=("agg_trade_id", "price", "quantity", "first_trade_id",
                 "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match"),
        timestamp_columns=("transact_time",),
    ),
    "perp_aggtrades": DatasetSpec(
        name="perp_aggtrades", market="futures/um", data_type="aggTrades", interval=None,
        granularities=("monthly", "daily"),
        columns=("agg_trade_id", "price", "quantity", "first_trade_id",
                 "last_trade_id", "transact_time", "is_buyer_maker"),
        timestamp_columns=("transact_time",),
    ),
}

# What a plain `backfill --datasets all` collects. liquidations excluded:
# Binance removed the liquidationSnapshot archive (June 2026); the spec stays
# for documentation, the forceOrder stream is the only live source.
DEFAULT_DATASETS = ("spot_klines", "perp_klines", "premium_index", "funding", "metrics")

# Delivery futures are collected for these underlyings only (Binance delivery
# liquidity is concentrated in BTC and ETH).
DELIVERY_UNDERLYINGS_UM = ("BTCUSDT", "ETHUSDT")
DELIVERY_UNDERLYINGS_CM = ("BTCUSD", "ETHUSD")
