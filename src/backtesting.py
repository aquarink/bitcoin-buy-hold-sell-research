from __future__ import annotations

import numpy as np
import pandas as pd


def shift_signals_for_execution(signals: pd.Series) -> pd.Series:
    return signals.shift(1)


def transaction_cost_rate(fee: float, slippage: float) -> float:
    return fee + slippage


def run_long_flat_backtest(
    df: pd.DataFrame,
    signal_col: str,
    open_col: str = "Open",
    fee: float = 0.0010,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    bt = df[["datetime_utc", open_col, "Close", signal_col]].copy()
    bt["executed_signal"] = shift_signals_for_execution(bt[signal_col])
    bt["position"] = 0
    current_position = 0
    positions: list[int] = []

    for signal in bt["executed_signal"]:
        if signal == "BUY" and current_position == 0:
            current_position = 1
        elif signal == "SELL" and current_position == 1:
            current_position = 0
        positions.append(current_position)

    bt["position"] = positions
    bt["asset_return"] = bt[open_col].pct_change().fillna(0.0)
    bt["position_prev"] = bt["position"].shift(1).fillna(0).astype(int)
    bt["strategy_return_gross"] = bt["position_prev"] * bt["asset_return"]
    bt["turnover"] = bt["position"].diff().abs().fillna(0).astype(float)
    bt["cost_rate"] = bt["turnover"] * transaction_cost_rate(fee=fee, slippage=slippage)
    bt["strategy_return_net"] = bt["strategy_return_gross"] - bt["cost_rate"]
    bt["equity_curve"] = (1 + bt["strategy_return_net"]).cumprod()
    bt["drawdown"] = bt["equity_curve"] / bt["equity_curve"].cummax() - 1
    return bt


def summarize_backtest(bt: pd.DataFrame) -> dict[str, float]:
    total_return = bt["equity_curve"].iloc[-1] - 1
    max_drawdown = bt["drawdown"].min()
    num_trades = int(bt["turnover"].sum())
    exposure_time = float(bt["position"].mean())
    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "num_trades": num_trades,
        "exposure_time": exposure_time,
    }
