from __future__ import annotations

import pandas as pd


BUY = "BUY"
HOLD = "HOLD"
SELL = "SELL"


def compute_future_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    return df["Close"].shift(-horizon).div(df["Close"]).sub(1.0)


def create_multiclass_labels(future_return: pd.Series, threshold: float) -> pd.Series:
    labels = pd.Series(HOLD, index=future_return.index, dtype="object")
    labels = labels.mask(future_return > threshold, BUY)
    labels = labels.mask(future_return < -threshold, SELL)
    labels = labels.mask(future_return.isna(), pd.NA)
    return labels


def append_label_columns(df: pd.DataFrame, horizons: list[int], thresholds: list[float]) -> pd.DataFrame:
    result = df.copy()
    for horizon in horizons:
        future_return = compute_future_return(result, horizon)
        result[f"future_return_{horizon}h"] = future_return
        for threshold in thresholds:
            threshold_slug = str(round(threshold * 10_000))
            result[f"label_h{horizon}_thr_{threshold_slug}bp"] = create_multiclass_labels(
                future_return=future_return,
                threshold=threshold,
            )
    return result


def class_distribution_by_year(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    temp = df[["datetime_utc", label_col]].dropna().copy()
    temp["year"] = temp["datetime_utc"].dt.year
    distribution = (
        temp.groupby(["year", label_col])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="year", columns=label_col, values="count")
        .fillna(0)
        .reset_index()
    )
    for label in [BUY, HOLD, SELL]:
        if label not in distribution.columns:
            distribution[label] = 0
    distribution["total"] = distribution[[BUY, HOLD, SELL]].sum(axis=1)
    for label in [BUY, HOLD, SELL]:
        distribution[f"{label.lower()}_pct"] = distribution[label] / distribution["total"]
    return distribution.sort_values("year").reset_index(drop=True)
