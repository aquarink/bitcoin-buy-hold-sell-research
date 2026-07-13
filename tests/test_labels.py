from __future__ import annotations

import pandas as pd

from src.labels import BUY, HOLD, SELL, compute_future_return, create_multiclass_labels


def test_compute_future_return() -> None:
    df = pd.DataFrame({"Close": [100.0, 110.0, 121.0]})
    result = compute_future_return(df, horizon=1)
    assert round(result.iloc[0], 4) == 0.1
    assert round(result.iloc[1], 4) == 0.1
    assert pd.isna(result.iloc[2])


def test_create_multiclass_labels() -> None:
    future_return = pd.Series([0.01, 0.001, -0.02, None])
    labels = create_multiclass_labels(future_return, threshold=0.005)
    assert labels.iloc[0] == BUY
    assert labels.iloc[1] == HOLD
    assert labels.iloc[2] == SELL
    assert pd.isna(labels.iloc[3])
