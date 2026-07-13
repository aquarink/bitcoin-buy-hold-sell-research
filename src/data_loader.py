from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


CSV_DTYPES = {
    "Timestamp": "int64",
    "Open": "float32",
    "High": "float32",
    "Low": "float32",
    "Close": "float32",
    "Volume": "float32",
}


def read_minute_csv_in_chunks(path: str | Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, dtype=CSV_DTYPES, chunksize=chunksize)


def read_minute_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=CSV_DTYPES)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
