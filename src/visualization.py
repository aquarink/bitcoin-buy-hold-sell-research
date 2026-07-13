from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_hourly_eda(hourly_df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = hourly_df.copy()
    df["simple_return"] = df["Close"].pct_change()
    df["volatility_24h"] = df["simple_return"].rolling(24, min_periods=24).std()
    yearly_counts = df["datetime_utc"].dt.year.value_counts().sort_index()

    figures: list[Path] = []

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["datetime_utc"], df["Close"], linewidth=0.8)
    ax.set_title("BTCUSD 1H Close Price")
    ax.set_ylabel("Price")
    price_path = output_dir / "price_1h.png"
    fig.tight_layout()
    fig.savefig(price_path, dpi=150)
    plt.close(fig)
    figures.append(price_path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["datetime_utc"], df["Volume"], linewidth=0.6, color="tab:orange")
    ax.set_title("BTCUSD 1H Volume")
    ax.set_ylabel("Volume")
    volume_path = output_dir / "volume_1h.png"
    fig.tight_layout()
    fig.savefig(volume_path, dpi=150)
    plt.close(fig)
    figures.append(volume_path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["datetime_utc"], df["simple_return"], linewidth=0.5, color="tab:green")
    ax.set_title("BTCUSD 1H Simple Return")
    ax.set_ylabel("Return")
    return_path = output_dir / "return_1h.png"
    fig.tight_layout()
    fig.savefig(return_path, dpi=150)
    plt.close(fig)
    figures.append(return_path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["datetime_utc"], df["volatility_24h"], linewidth=0.8, color="tab:red")
    ax.set_title("BTCUSD 24H Rolling Volatility")
    ax.set_ylabel("Volatility")
    vol_path = output_dir / "volatility_24h.png"
    fig.tight_layout()
    fig.savefig(vol_path, dpi=150)
    plt.close(fig)
    figures.append(vol_path)

    fig, ax = plt.subplots(figsize=(12, 4))
    yearly_counts.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title("Hourly Candle Count per Year")
    ax.set_ylabel("Candles")
    count_path = output_dir / "hourly_candles_per_year.png"
    fig.tight_layout()
    fig.savefig(count_path, dpi=150)
    plt.close(fig)
    figures.append(count_path)
    return figures
