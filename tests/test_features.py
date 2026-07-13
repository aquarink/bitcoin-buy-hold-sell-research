from __future__ import annotations

import pandas as pd

from src.features import build_features


def test_rolling_features_use_only_current_and_past_values() -> None:
    datetimes = pd.date_range("2026-01-01 00:00:00+00:00", periods=12, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "datetime_utc": datetimes,
            "Open": [float(i) for i in range(1, 13)],
            "High": [float(i) + 1 for i in range(1, 13)],
            "Low": [float(i) - 1 for i in range(1, 13)],
            "Close": [float(i) for i in range(1, 13)],
            "Volume": [10.0] * 12,
            "timestamp": [int(ts.timestamp()) for ts in datetimes],
        }
    )
    features = build_features(df)
    assert features.loc[9, "sma_10"] == 5.5
    assert features.loc[10, "sma_10"] == 6.5
