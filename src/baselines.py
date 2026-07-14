from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import BUY, HOLD, SELL


CLASS_ORDER = [BUY, HOLD, SELL]


@dataclass(frozen=True)
class BaselinePrediction:
    predicted_label: pd.Series
    probabilities: pd.DataFrame


def _probabilities_from_labels(labels: pd.Series) -> pd.DataFrame:
    probs = pd.DataFrame(0.0, index=labels.index, columns=[f"probability_{c.lower()}" for c in CLASS_ORDER])
    for label in CLASS_ORDER:
        probs.loc[labels == label, f"probability_{label.lower()}"] = 1.0
    return probs


def always_hold(index: pd.Index) -> BaselinePrediction:
    labels = pd.Series(HOLD, index=index, dtype="object")
    return BaselinePrediction(predicted_label=labels, probabilities=_probabilities_from_labels(labels))


def always_buy(index: pd.Index) -> BaselinePrediction:
    labels = pd.Series(BUY, index=index, dtype="object")
    return BaselinePrediction(predicted_label=labels, probabilities=_probabilities_from_labels(labels))


def majority_class(train_labels: pd.Series, index: pd.Index) -> BaselinePrediction:
    majority = train_labels.dropna().value_counts().idxmax()
    labels = pd.Series(majority, index=index, dtype="object")
    return BaselinePrediction(predicted_label=labels, probabilities=_probabilities_from_labels(labels))


def macd_crossover_strategy(df: pd.DataFrame) -> BaselinePrediction:
    labels = pd.Series(HOLD, index=df.index, dtype="object")
    bullish_cross = (df["macd_line"] > df["macd_signal"]) & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
    bearish_cross = (df["macd_line"] < df["macd_signal"]) & (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
    labels = labels.mask(bullish_cross, BUY)
    labels = labels.mask(bearish_cross, SELL)
    return BaselinePrediction(predicted_label=labels, probabilities=_probabilities_from_labels(labels))


def bollinger_band_strategy(df: pd.DataFrame) -> BaselinePrediction:
    labels = pd.Series(HOLD, index=df.index, dtype="object")
    buy_signal = (df["Close"] < df["bb_lower"]) & (df["Close"].shift(1) <= df["bb_lower"].shift(1))
    sell_signal = (df["Close"] > df["bb_upper"]) & (df["Close"].shift(1) >= df["bb_upper"].shift(1))
    labels = labels.mask(buy_signal, BUY)
    labels = labels.mask(sell_signal, SELL)
    return BaselinePrediction(predicted_label=labels, probabilities=_probabilities_from_labels(labels))


def combined_macd_bollinger_strategy(df: pd.DataFrame) -> BaselinePrediction:
    labels = pd.Series(HOLD, index=df.index, dtype="object")
    buy_signal = (df["Close"] < df["bb_lower"]) & (df["macd_line"] > df["macd_signal"])
    sell_signal = (df["Close"] > df["bb_upper"]) & (df["macd_line"] < df["macd_signal"])
    labels = labels.mask(buy_signal, BUY)
    labels = labels.mask(sell_signal, SELL)
    return BaselinePrediction(predicted_label=labels, probabilities=_probabilities_from_labels(labels))
