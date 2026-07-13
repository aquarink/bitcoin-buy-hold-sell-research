from __future__ import annotations

import pandas as pd

from src.preprocessing import aggregate_ohlcv, unix_to_utc, validate_strictly_sorted


def test_unix_to_utc_conversion() -> None:
    timestamps = pd.Series([1325376060, 1783216500])
    result = unix_to_utc(timestamps)
    assert str(result.iloc[0]) == "2012-01-01 00:01:00+00:00"
    assert str(result.iloc[1]) == "2026-07-05 01:55:00+00:00"


def test_validate_strictly_sorted_rejects_duplicate_timestamp() -> None:
    df = pd.DataFrame({"Timestamp": [1, 2, 2]})
    try:
        validate_strictly_sorted(df)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for duplicate timestamp.")


def test_aggregate_ohlcv_hourly() -> None:
    index = pd.to_datetime(
        [
            "2026-01-01 00:00:00+00:00",
            "2026-01-01 00:30:00+00:00",
            "2026-01-01 00:59:00+00:00",
            "2026-01-01 01:00:00+00:00",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "Open": [10, 11, 12, 20],
            "High": [11, 12, 15, 21],
            "Low": [9, 10, 11, 19],
            "Close": [10.5, 11.5, 14.0, 20.5],
            "Volume": [1, 2, 3, 4],
        },
        index=index,
    )
    result = aggregate_ohlcv(df)
    first_row = result.iloc[0]
    assert first_row["Open"] == 10
    assert first_row["High"] == 15
    assert first_row["Low"] == 9
    assert first_row["Close"] == 14.0
    assert first_row["Volume"] == 6
