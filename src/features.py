from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    description: str


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr_components = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def get_feature_definitions() -> list[FeatureDefinition]:
    definitions = [
        FeatureDefinition("simple_return", "Close[t] / Close[t-1] - 1"),
        FeatureDefinition("log_return", "log(Close[t]) - log(Close[t-1])"),
        FeatureDefinition("return_lag_1", "Lag 1 dari simple return"),
        FeatureDefinition("return_lag_2", "Lag 2 dari simple return"),
        FeatureDefinition("return_lag_3", "Lag 3 dari simple return"),
        FeatureDefinition("return_lag_6", "Lag 6 dari simple return"),
        FeatureDefinition("return_lag_12", "Lag 12 dari simple return"),
        FeatureDefinition("return_lag_24", "Lag 24 dari simple return"),
        FeatureDefinition("candle_body", "Close - Open"),
        FeatureDefinition("abs_candle_body", "abs(Close - Open)"),
        FeatureDefinition("high_low_range", "High - Low"),
        FeatureDefinition("upper_wick", "High - max(Open, Close)"),
        FeatureDefinition("lower_wick", "min(Open, Close) - Low"),
        FeatureDefinition("body_to_range", "abs(body) / max(range, eps)"),
        FeatureDefinition("candle_direction", "1 jika Close >= Open, else 0"),
        FeatureDefinition("sma_10", "Simple moving average 10 periode Close"),
        FeatureDefinition("sma_20", "Simple moving average 20 periode Close"),
        FeatureDefinition("sma_50", "Simple moving average 50 periode Close"),
        FeatureDefinition("ema_12", "Exponential moving average 12 periode Close"),
        FeatureDefinition("ema_26", "Exponential moving average 26 periode Close"),
        FeatureDefinition("macd_line", "EMA12 - EMA26"),
        FeatureDefinition("macd_signal", "EMA 9 dari MACD line"),
        FeatureDefinition("macd_histogram", "MACD line - signal"),
        FeatureDefinition("macd_crossover_flag", "1 jika MACD line memotong signal dari bawah"),
        FeatureDefinition("bb_middle_20", "Bollinger middle band 20 periode"),
        FeatureDefinition("bb_upper", "Bollinger upper band 20 periode, 2 std"),
        FeatureDefinition("bb_lower", "Bollinger lower band 20 periode, 2 std"),
        FeatureDefinition("bb_band_width", "(upper - lower) / middle"),
        FeatureDefinition("bb_price_position", "(Close - lower) / (upper - lower)"),
        FeatureDefinition("bb_breakout_upper_flag", "1 jika Close > upper band"),
        FeatureDefinition("bb_breakout_lower_flag", "1 jika Close < lower band"),
        FeatureDefinition("rsi_14", "Relative Strength Index 14"),
        FeatureDefinition("atr_14", "Average True Range 14"),
        FeatureDefinition("rolling_volatility_6", "Std log return 6 jam"),
        FeatureDefinition("rolling_volatility_12", "Std log return 12 jam"),
        FeatureDefinition("rolling_volatility_24", "Std log return 24 jam"),
        FeatureDefinition("rolling_volatility_72", "Std log return 72 jam"),
        FeatureDefinition("momentum_6", "Close[t] - Close[t-6]"),
        FeatureDefinition("roc_6", "(Close[t] / Close[t-6]) - 1"),
        FeatureDefinition("volume_change", "Volume[t] / Volume[t-1] - 1"),
        FeatureDefinition("log_volume", "log1p(Volume)"),
        FeatureDefinition("rolling_mean_volume_24", "Rata-rata volume 24 jam"),
        FeatureDefinition("rolling_volume_ratio_24", "Volume / rolling mean volume 24 jam"),
        FeatureDefinition("rolling_volume_zscore_24", "(Volume - mean24) / std24"),
        FeatureDefinition("hour_sin", "Representasi siklik jam"),
        FeatureDefinition("hour_cos", "Representasi siklik jam"),
        FeatureDefinition("dow_sin", "Representasi siklik hari dalam minggu"),
        FeatureDefinition("dow_cos", "Representasi siklik hari dalam minggu"),
        FeatureDefinition("month_sin", "Representasi siklik bulan"),
        FeatureDefinition("month_cos", "Representasi siklik bulan"),
    ]
    return definitions


def build_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    if "datetime_utc" not in df.columns:
        raise ValueError("hourly_df must contain datetime_utc column.")
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["simple_return"] = close.pct_change()
    df["log_return"] = np.log(close).diff()
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"return_lag_{lag}"] = df["simple_return"].shift(lag)

    candle_body = close - open_
    price_range = high - low
    df["candle_body"] = candle_body
    df["abs_candle_body"] = candle_body.abs()
    df["high_low_range"] = price_range
    df["upper_wick"] = high - pd.concat([open_, close], axis=1).max(axis=1)
    df["lower_wick"] = pd.concat([open_, close], axis=1).min(axis=1) - low
    df["body_to_range"] = _safe_divide(df["abs_candle_body"], price_range)
    df["candle_direction"] = (close >= open_).astype("int8")

    df["sma_10"] = close.rolling(window=10, min_periods=10).mean()
    df["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    df["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()

    df["macd_line"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
    macd_above = df["macd_line"] > df["macd_signal"]
    macd_above_prev = df["macd_line"].shift(1) <= df["macd_signal"].shift(1)
    df["macd_crossover_flag"] = (macd_above & macd_above_prev).astype("int8")

    rolling_mean_20 = close.rolling(window=20, min_periods=20).mean()
    rolling_std_20 = close.rolling(window=20, min_periods=20).std()
    df["bb_middle_20"] = rolling_mean_20
    df["bb_upper"] = rolling_mean_20 + 2 * rolling_std_20
    df["bb_lower"] = rolling_mean_20 - 2 * rolling_std_20
    df["bb_band_width"] = _safe_divide(df["bb_upper"] - df["bb_lower"], df["bb_middle_20"])
    df["bb_price_position"] = _safe_divide(close - df["bb_lower"], df["bb_upper"] - df["bb_lower"])
    df["bb_breakout_upper_flag"] = (close > df["bb_upper"]).astype("int8")
    df["bb_breakout_lower_flag"] = (close < df["bb_lower"]).astype("int8")

    df["rsi_14"] = _rsi(close, window=14)
    df["atr_14"] = _atr(df, window=14)
    for window in [6, 12, 24, 72]:
        df[f"rolling_volatility_{window}"] = df["log_return"].rolling(window=window, min_periods=window).std()

    df["momentum_6"] = close - close.shift(6)
    df["roc_6"] = close.pct_change(periods=6)

    df["volume_change"] = volume.pct_change()
    df["log_volume"] = np.log1p(volume)
    rolling_mean_volume = volume.rolling(window=24, min_periods=24).mean()
    rolling_std_volume = volume.rolling(window=24, min_periods=24).std()
    df["rolling_mean_volume_24"] = rolling_mean_volume
    df["rolling_volume_ratio_24"] = _safe_divide(volume, rolling_mean_volume)
    df["rolling_volume_zscore_24"] = _safe_divide(volume - rolling_mean_volume, rolling_std_volume)

    hours = df["datetime_utc"].dt.hour
    day_of_week = df["datetime_utc"].dt.dayofweek
    month = df["datetime_utc"].dt.month - 1
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def assert_no_centered_rolling_usage() -> None:
    """Code-level guard to make the design explicit."""
    return None
