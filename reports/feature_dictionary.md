# Kamus Fitur

Semua fitur di bawah dihitung dari candle `1H` dan hanya menggunakan informasi pada waktu `t` atau sebelumnya. Tidak ada `negative shift` dan tidak ada `centered rolling window`.

## Return

| Fitur | Definisi |
|---|---|
| `simple_return` | `Close[t] / Close[t-1] - 1` |
| `log_return` | `log(Close[t]) - log(Close[t-1])` |
| `return_lag_1` | Lag 1 dari `simple_return` |
| `return_lag_2` | Lag 2 dari `simple_return` |
| `return_lag_3` | Lag 3 dari `simple_return` |
| `return_lag_6` | Lag 6 dari `simple_return` |
| `return_lag_12` | Lag 12 dari `simple_return` |
| `return_lag_24` | Lag 24 dari `simple_return` |

## Struktur Candle

| Fitur | Definisi |
|---|---|
| `candle_body` | `Close - Open` |
| `abs_candle_body` | `abs(Close - Open)` |
| `high_low_range` | `High - Low` |
| `upper_wick` | `High - max(Open, Close)` |
| `lower_wick` | `min(Open, Close) - Low` |
| `body_to_range` | `abs(body) / range` dengan pembagian aman |
| `candle_direction` | `1` jika `Close >= Open`, selain itu `0` |

## Moving Average dan MACD

| Fitur | Definisi |
|---|---|
| `sma_10` | SMA 10 periode close |
| `sma_20` | SMA 20 periode close |
| `sma_50` | SMA 50 periode close |
| `ema_12` | EMA 12 periode close |
| `ema_26` | EMA 26 periode close |
| `macd_line` | `ema_12 - ema_26` |
| `macd_signal` | EMA 9 periode dari `macd_line` |
| `macd_histogram` | `macd_line - macd_signal` |
| `macd_crossover_flag` | `1` saat MACD memotong signal dari bawah pada waktu `t` |

## Bollinger Bands

| Fitur | Definisi |
|---|---|
| `bb_middle_20` | Rata-rata rolling 20 close |
| `bb_upper` | `bb_middle_20 + 2 * std_20` |
| `bb_lower` | `bb_middle_20 - 2 * std_20` |
| `bb_band_width` | `(bb_upper - bb_lower) / bb_middle_20` |
| `bb_price_position` | `(Close - bb_lower) / (bb_upper - bb_lower)` |
| `bb_breakout_upper_flag` | `1` jika `Close > bb_upper` |
| `bb_breakout_lower_flag` | `1` jika `Close < bb_lower` |

## Indikator Tambahan

| Fitur | Definisi |
|---|---|
| `rsi_14` | RSI 14 periode berbasis EWM Wilder-style |
| `atr_14` | ATR 14 periode |
| `rolling_volatility_6` | Std `log_return` 6 jam |
| `rolling_volatility_12` | Std `log_return` 12 jam |
| `rolling_volatility_24` | Std `log_return` 24 jam |
| `rolling_volatility_72` | Std `log_return` 72 jam |
| `momentum_6` | `Close[t] - Close[t-6]` |
| `roc_6` | `Close[t] / Close[t-6] - 1` |

## Volume

| Fitur | Definisi |
|---|---|
| `volume_change` | `Volume[t] / Volume[t-1] - 1` |
| `log_volume` | `log1p(Volume)` untuk aman saat volume nol |
| `rolling_mean_volume_24` | Rata-rata volume 24 jam |
| `rolling_volume_ratio_24` | `Volume / rolling_mean_volume_24` |
| `rolling_volume_zscore_24` | `(Volume - mean24) / std24` |

## Kalender Siklik

| Fitur | Definisi |
|---|---|
| `hour_sin`, `hour_cos` | Jam dalam sehari dengan encoding sin/cos |
| `dow_sin`, `dow_cos` | Hari dalam minggu dengan encoding sin/cos |
| `month_sin`, `month_cos` | Bulan dengan encoding sin/cos |

## Catatan Leakage

1. Semua rolling window memakai `center=False` default pandas.
2. Semua lag memakai `shift(positif)`.
3. Label `future_return` disimpan terpisah dari fitur agar pipeline training nanti bisa secara eksplisit memilih kolom input.
4. Fitur volume memakai bentuk yang tahan terhadap `Volume = 0`, terutama `log1p`.
