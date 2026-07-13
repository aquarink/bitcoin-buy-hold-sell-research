from __future__ import annotations

import pandas as pd

from src.temporal_split import build_fold_mask, final_holdout_mask, generate_expanding_folds, generate_sliding_folds


def test_expanding_and_sliding_fold_generation() -> None:
    expanding = generate_expanding_folds(start_train_year=2015, first_validation_year=2018, last_test_year=2020)
    sliding = generate_sliding_folds(first_train_year=2015, last_test_year=2021, train_window_years=4)
    assert expanding[0].validation_start.year == 2018
    assert expanding[0].test_start.year == 2019
    assert sliding[0].train_start.year == 2015
    assert sliding[0].test_start.year == 2020


def test_no_overlap_between_train_validation_test() -> None:
    df = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2018-06-01", "2019-06-01", "2020-06-01"],
                utc=True,
            )
        }
    )
    fold = generate_expanding_folds(start_train_year=2018, first_validation_year=2018, last_test_year=2019)[0]
    masks = build_fold_mask(df, fold)
    assert not (masks["train"] & masks["validation"]).any()
    assert not (masks["train"] & masks["test"]).any()
    assert not (masks["validation"] & masks["test"]).any()


def test_final_holdout_mask_uses_q1_2026() -> None:
    df = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2025-12-31 23:00:00+00:00", "2026-02-01 00:00:00+00:00", "2026-04-01 00:00:00+00:00"],
                utc=True,
            )
        }
    )
    mask = final_holdout_mask(df)
    assert mask.tolist() == [False, True, False]
