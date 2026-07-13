from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import read_minute_csv


EXPECTED_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True)
class ResampleArtifacts:
    hourly: pd.DataFrame
    yearly_stats: pd.DataFrame


def unix_to_utc(timestamp_series: pd.Series) -> pd.Series:
    return pd.to_datetime(timestamp_series, unit="s", utc=True)


def validate_input_columns(df: pd.DataFrame) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")


def validate_strictly_sorted(df: pd.DataFrame, timestamp_col: str = "Timestamp") -> None:
    diffs = df[timestamp_col].diff().dropna()
    if (diffs <= 0).any():
        raise ValueError("Timestamps must be strictly increasing with no duplicates.")


def add_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["datetime_utc"] = unix_to_utc(result["Timestamp"])
    return result.set_index("datetime_utc")


def aggregate_ohlcv(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware UTC.")

    hourly = (
        df.resample(timeframe, label="left", closed="left")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    hourly.index.name = "datetime_utc"
    hourly["timestamp"] = (hourly.index.view("int64") // 10**9).astype("int64")
    return hourly.reset_index()


def compute_hourly_yearly_stats(hourly_df: pd.DataFrame) -> pd.DataFrame:
    stats = hourly_df.copy()
    stats["year"] = stats["datetime_utc"].dt.year
    stats["simple_return"] = stats["Close"].pct_change()
    stats["log_return"] = np.log(stats["Close"]).diff()
    stats["range"] = stats["High"] - stats["Low"]

    grouped = stats.groupby("year", as_index=False).agg(
        candles=("datetime_utc", "size"),
        zero_volume_candles=("Volume", lambda s: int((s == 0).sum())),
        avg_close=("Close", "mean"),
        avg_volume=("Volume", "mean"),
        return_volatility=("log_return", "std"),
        avg_range=("range", "mean"),
    )
    grouped["zero_volume_pct"] = grouped["zero_volume_candles"] / grouped["candles"]
    return grouped


def preprocess_minute_to_hourly(csv_path: str | Path, parquet_path: str | Path) -> ResampleArtifacts:
    minute_df = read_minute_csv(csv_path)
    validate_input_columns(minute_df)
    validate_strictly_sorted(minute_df, "Timestamp")
    indexed = add_datetime_index(minute_df)
    hourly = aggregate_ohlcv(indexed[EXPECTED_COLUMNS[1:]], timeframe="1h")
    yearly_stats = compute_hourly_yearly_stats(hourly)

    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(parquet_path, index=False)
    return ResampleArtifacts(hourly=hourly, yearly_stats=yearly_stats)
