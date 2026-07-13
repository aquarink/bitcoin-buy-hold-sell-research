from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    fold_name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_type: str


def _year_start(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=1, day=1, tz="UTC")


def _year_end(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=12, day=31, hour=23, tz="UTC")


def generate_expanding_folds(start_train_year: int, first_validation_year: int, last_test_year: int) -> list[TemporalFold]:
    folds: list[TemporalFold] = []
    for test_year in range(first_validation_year + 1, last_test_year + 1):
        validation_year = test_year - 1
        folds.append(
            TemporalFold(
                fold_name=f"expanding_test_{test_year}",
                train_start=_year_start(start_train_year),
                train_end=_year_end(validation_year - 1),
                validation_start=_year_start(validation_year),
                validation_end=_year_end(validation_year),
                test_start=_year_start(test_year),
                test_end=_year_end(test_year),
                window_type="expanding",
            )
        )
    return folds


def generate_sliding_folds(
    first_train_year: int,
    last_test_year: int,
    train_window_years: int = 4,
) -> list[TemporalFold]:
    folds: list[TemporalFold] = []
    for test_year in range(first_train_year + train_window_years + 1, last_test_year + 1):
        validation_year = test_year - 1
        train_start_year = validation_year - train_window_years
        train_end_year = validation_year - 1
        folds.append(
            TemporalFold(
                fold_name=f"sliding_test_{test_year}",
                train_start=_year_start(train_start_year),
                train_end=_year_end(train_end_year),
                validation_start=_year_start(validation_year),
                validation_end=_year_end(validation_year),
                test_start=_year_start(test_year),
                test_end=_year_end(test_year),
                window_type="sliding",
            )
        )
    return folds


def final_holdout_mask(df: pd.DataFrame, start: str = "2026-01-01T00:00:00Z", end: str = "2026-03-31T23:59:59Z") -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return (df["datetime_utc"] >= start_ts) & (df["datetime_utc"] <= end_ts)


def build_fold_mask(df: pd.DataFrame, fold: TemporalFold) -> dict[str, pd.Series]:
    train = (df["datetime_utc"] >= fold.train_start) & (df["datetime_utc"] <= fold.train_end)
    validation = (df["datetime_utc"] >= fold.validation_start) & (df["datetime_utc"] <= fold.validation_end)
    test = (df["datetime_utc"] >= fold.test_start) & (df["datetime_utc"] <= fold.test_end)
    overlap = (train & validation) | (train & test) | (validation & test)
    if overlap.any():
        raise ValueError(f"Fold {fold.fold_name} contains overlapping splits.")
    return {"train": train, "validation": validation, "test": test}
