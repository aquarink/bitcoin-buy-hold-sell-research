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
    bt = bt.reset_index(drop=True)
    returns = bt["strategy_return_net"].fillna(0.0)
    total_return = bt["equity_curve"].iloc[-1] - 1
    max_drawdown = bt["drawdown"].min()
    num_trades = int(bt["turnover"].sum())
    exposure_time = float(bt["position"].mean())
    hours_per_year = 24 * 365
    if len(bt) > 0:
        annualized_return = float((bt["equity_curve"].iloc[-1] ** (hours_per_year / len(bt))) - 1)
    else:
        annualized_return = 0.0
    volatility = float(returns.std(ddof=0))
    downside = returns.where(returns < 0, 0.0)
    downside_std = float(downside.std(ddof=0))
    sharpe = float(np.sqrt(hours_per_year) * returns.mean() / volatility) if volatility > 0 else 0.0
    sortino = float(np.sqrt(hours_per_year) * returns.mean() / downside_std) if downside_std > 0 else 0.0
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    positive_returns = returns[returns > 0].sum()
    negative_returns = returns[returns < 0].abs().sum()
    profit_factor = float(positive_returns / negative_returns) if negative_returns > 0 else 0.0
    entry_mask = ((bt["position_prev"] == 0) & (bt["position"] == 1)).astype(int)
    trade_id = entry_mask.cumsum()
    in_position = bt["position"] == 1
    trade_returns_series = (
        (1.0 + bt.loc[in_position, "strategy_return_net"])
        .groupby(trade_id.loc[in_position])
        .prod()
        .sub(1.0)
    )
    trade_count = int(trade_returns_series.shape[0])
    win_rate = float((trade_returns_series > 0).mean()) if trade_count > 0 else 0.0
    avg_return_per_trade = float(trade_returns_series.mean()) if trade_count > 0 else 0.0
    return {
        "total_return": float(total_return),
        "cumulative_return": float(total_return),
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return_per_trade": avg_return_per_trade,
        "num_trades": trade_count,
        "turnover": float(bt["turnover"].sum()),
        "exposure_time": exposure_time,
        "return_after_costs": float(returns.sum()),
    }
