from __future__ import annotations

import pandas as pd

from src.backtesting import run_long_flat_backtest, shift_signals_for_execution, transaction_cost_rate


def test_signal_is_shifted_to_next_candle() -> None:
    signals = pd.Series(["BUY", "HOLD", "SELL"])
    shifted = shift_signals_for_execution(signals)
    assert pd.isna(shifted.iloc[0])
    assert shifted.iloc[1] == "BUY"


def test_transaction_cost_rate() -> None:
    assert transaction_cost_rate(0.0010, 0.0005) == 0.0015


def test_backtest_does_not_trade_same_candle_signal() -> None:
    df = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                ["2026-01-01 00:00:00+00:00", "2026-01-01 01:00:00+00:00", "2026-01-01 02:00:00+00:00"],
                utc=True,
            ),
            "Open": [100.0, 110.0, 121.0],
            "Close": [100.0, 110.0, 121.0],
            "signal": ["BUY", "HOLD", "SELL"],
        }
    )
    bt = run_long_flat_backtest(df, signal_col="signal")
    assert bt["position"].tolist() == [0, 1, 1]
