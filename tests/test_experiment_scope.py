from __future__ import annotations

import pandas as pd

from src.experiment_scope import apply_experiment_min_datetime


def test_apply_experiment_min_datetime_filters_older_rows() -> None:
    df = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2014-12-31 23:00:00+00:00", "2015-01-01 00:00:00+00:00", "2015-01-01 01:00:00+00:00"],
                utc=True,
            ),
            "value": [1, 2, 3],
        }
    )
    result = apply_experiment_min_datetime(df, "2015-01-01T00:00:00Z")
    assert result["value"].tolist() == [2, 3]
