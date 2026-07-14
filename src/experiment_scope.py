from __future__ import annotations

import pandas as pd


def apply_experiment_min_datetime(df: pd.DataFrame, min_datetime: str) -> pd.DataFrame:
    min_ts = pd.Timestamp(min_datetime)
    result = df[df["datetime_utc"] >= min_ts].copy()
    return result.reset_index(drop=True)
