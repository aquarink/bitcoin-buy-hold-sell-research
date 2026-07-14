# Hasil Eksperimen Sementara

## 1. Tujuan Eksperimen

Tahap ini bertujuan membangun baseline dan model klasik untuk tugas klasifikasi `BUY/HOLD/SELL` Bitcoin pada timeframe `1 jam`, dengan:

- horizon utama `6 jam`;
- walk-forward validation `expanding window`;
- final holdout `2026 Q1` tetap tidak digunakan.

Tahap ini **belum** mencakup `LSTM`, `sliding window`, `ablation`, dan `final test 2026 Q1`.

## 2. Deskripsi Dataset

- Sumber data mentah: `btcusd_1-min_data.csv`
- Rentang data mentah aktual: `2012-01-01 00:01:00 UTC` sampai `2026-07-05 01:55:00 UTC`
- Hasil resampling utama: `127,178` candle `1H`
- Tidak ada missing hour setelah resampling
- Eksperimen model pada Tahap 3 dibatasi ke data `>= 2015-01-01 UTC`

Audit lengkap ada di [data_audit.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/data_audit.md).

## 3. Rentang Tanggal Aktual

### Data mentah

| Minimum | Maximum |
|---|---|
| `2012-01-01 00:01:00 UTC` | `2026-07-05 01:55:00 UTC` |

### Data 1 jam

| Minimum | Maximum |
|---|---|
| `2012-01-01 00:00:00 UTC` | `2026-07-05 01:00:00 UTC` |

## 4. Hasil Audit Data

Temuan penting:

- tidak ada missing value pada data mentah;
- tidak ada gap timestamp 1 menit;
- tidak ada timestamp duplikat;
- aturan dasar OHLC valid;
- volume nol sangat besar pada tahun-tahun awal, terutama 2012.

Setelah agregasi ke 1 jam, masalah volume nol jauh berkurang, tetapi `2012` masih memiliki proporsi `37.37%` zero-volume hourly candle.

## 5. Metode Preprocessing

- Resampling `1m -> 1h`
- Aturan agregasi:
  - `Open = first`
  - `High = max`
  - `Low = min`
  - `Close = last`
  - `Volume = sum`
- Output utama:
  - `data/processed/btcusd_1h.parquet`
  - `data/processed/btcusd_1h_features.parquet`
  - `data/processed/btcusd_1h_features_labels.parquet`

## 6. Fitur

Fitur mencakup:

- return dan lag return;
- struktur candle;
- moving average;
- MACD;
- Bollinger Bands;
- RSI;
- ATR;
- rolling volatility;
- momentum dan ROC;
- fitur volume;
- fitur kalender sin/cos.

Definisi rinci ada di [feature_dictionary.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/feature_dictionary.md).

## 7. Label

Setup label tahap ini:

- horizon utama: `6 jam`
- kandidat threshold: `0.25%`, `0.50%`, `0.75%`, `1.00%`

Formula:

```text
future_return_t_6h = (Close[t+6] - Close[t]) / Close[t]
```

Threshold final tahap ini dipilih dari validation expanding window pra-2026, bukan dari final holdout.

Dokumentasi label ada di [label_documentation.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/label_documentation.md).

## 8. Walk-Forward Validation

Skema yang dijalankan pada Tahap 3:

- `Expanding window`
- train mulai `2015`
- validation mulai `2018`
- fold test berjalan dari `2019` sampai `2025`

Fold:

| Fold | Train | Validation | Test |
|---|---|---|---|
| `expanding_test_2019` | 2015-2017 | 2018 | 2019 |
| `expanding_test_2020` | 2015-2018 | 2019 | 2020 |
| `expanding_test_2021` | 2015-2019 | 2020 | 2021 |
| `expanding_test_2022` | 2015-2020 | 2021 | 2022 |
| `expanding_test_2023` | 2015-2021 | 2022 | 2023 |
| `expanding_test_2024` | 2015-2022 | 2023 | 2024 |
| `expanding_test_2025` | 2015-2023 | 2024 | 2025 |

## 9. Model

Model yang dijalankan:

- `always_hold`
- `majority_class`
- `MACD`
- `Bollinger Bands`
- `MACD + Bollinger Bands`
- `Logistic Regression`
- `XGBoost`

LSTM belum dijalankan pada tahap ini.

## 10. Seleksi Threshold

Hasil seleksi threshold validation expanding:

| Threshold | Mean Macro F1 | Std Macro F1 | Mean Balanced Accuracy | Mean Validation Backtest Return | Mean Trades |
|---:|---:|---:|---:|---:|---:|
| 0.50% | 0.3647 | 0.0337 | 0.3887 | -0.6264 | 1198.29 |
| 0.75% | 0.3575 | 0.0420 | 0.3901 | -0.5101 | 988.14 |
| 0.25% | 0.3563 | 0.0218 | 0.3864 | -0.7762 | 1410.29 |
| 1.00% | 0.3503 | 0.0538 | 0.3989 | -0.4151 | 807.71 |

Threshold yang dipilih untuk Tahap 3:

- `horizon = 6 jam`
- `threshold = 0.50%`

Alasan:

- memiliki `mean Macro F1` validation tertinggi;
- distribusi kelas lebih masuk akal dibanding threshold ekstrem;
- jumlah transaksi validation masih realistis walaupun hasil backtest validation belum baik.

## 11. Metrik Klasifikasi

### Rata-rata test per model antar-fold

| Model | Mean Macro F1 | Std Macro F1 | Mean Balanced Accuracy | Mean Accuracy | Mean MCC |
|---|---:|---:|---:|---:|---:|
| always_hold | 0.2031 | 0.0313 | 0.3333 | 0.4437 | 0.0000 |
| majority_class | 0.2031 | 0.0313 | 0.3333 | 0.4437 | 0.0000 |
| bollinger | 0.2398 | 0.0360 | 0.3375 | 0.4381 | rendah |
| macd | 0.2444 | 0.0275 | 0.3354 | 0.4337 | rendah |
| macd_bollinger | 0.2072 | 0.0320 | 0.3343 | 0.4439 | rendah |
| logistic_regression | 0.3100 | 0.0767 | 0.3703 | 0.3766 | moderat-rendah |
| xgboost | 0.3302 | 0.0542 | 0.3842 | 0.3636 | moderat-rendah |

Catatan:

- `XGBoost` unggul pada `Macro F1` rata-rata test.
- `Logistic Regression` lebih baik pada beberapa fold tertentu, tetapi kurang stabil.
- Accuracy model tidak tinggi, tetapi accuracy memang bukan target utama.

### Hasil per fold untuk model utama

| Fold | Logistic Macro F1 | XGBoost Macro F1 |
|---|---:|---:|
| 2019 | 0.3320 | 0.3357 |
| 2020 | 0.3871 | 0.3682 |
| 2021 | 0.2452 | 0.2225 |
| 2022 | 0.3989 | 0.3184 |
| 2023 | 0.3495 | 0.3947 |
| 2024 | 0.2599 | 0.3446 |
| 2025 | 0.1975 | 0.3269 |

Interpretasi:

- performa antar-tahun tidak stabil;
- tidak ada model yang dominan di semua fold;
- `2025` sulit untuk Logistic Regression;
- `2023-2025` relatif lebih baik untuk XGBoost dibanding Logistic Regression.

### Distribusi prediksi dan confidence

- Logistic Regression dan XGBoost memiliki confidence yang mirip:
  - mean confidence sekitar `0.469`
  - median sekitar `0.45`
- Ini menunjukkan model belum menghasilkan probabilitas yang sangat tegas.

## 12. Hasil Backtesting

Backtesting utama:

- long/flat spot
- eksekusi pada candle berikutnya
- fee `0.10%`
- slippage `0.05%`

### Ringkasan backtest test gabungan antar-fold

| Model | Total Return | Annualized Return | Sharpe | Max Drawdown | Win Rate | Trades |
|---|---:|---:|---:|---:|---:|---:|
| buy_and_hold | 22.6802 | 0.5710 | 1.0223 | -0.7724 | 1.0000 | 1 |
| macd_bollinger | 1.2331 | 0.1215 | 0.4919 | -0.4182 | 0.6383 | 47 |
| bollinger | 0.0117 | 0.0017 | 0.2466 | -0.7329 | 0.6350 | 411 |
| always_hold | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| majority_class | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| macd | -0.9889 | -0.4741 | -1.2054 | -0.9931 | 0.3051 | 2327 |
| logistic_regression | -0.9959 | -0.5432 | -1.9366 | -0.9961 | 0.3745 | 3004 |
| xgboost | -0.9999 | -0.7523 | -3.2318 | -0.9999 | 0.3934 | 3887 |

Interpretasi penting:

1. `Buy and hold` jauh mengungguli semua strategi aktif pada periode test gabungan 2019-2025.
2. `MACD + Bollinger` adalah strategi aktif terbaik pada setup ini.
3. `Logistic Regression` dan `XGBoost` menghasilkan klasifikasi yang lebih baik daripada baseline naif, tetapi performa trading setelah biaya transaksi sangat buruk.
4. Ini berarti peningkatan metrik klasifikasi **tidak otomatis** berubah menjadi strategi trading yang layak.

## 13. Expanding vs Sliding Window

Belum dijalankan pada Tahap 3. Perbandingan ini ditunda ke Tahap 5.

## 14. XGBoost vs LSTM

Belum bisa dibandingkan karena `LSTM` belum dijalankan pada Tahap 3.

## 15. Analisis Kondisi Pasar

Belum dijalankan pada Tahap 3.

## 16. Ablation Study

Belum dijalankan pada Tahap 3.

## 17. Final Test 2026 Q1

Belum dijalankan.

Semua data `2026 Q1` tetap disisihkan sebagai final holdout dan belum dipakai untuk:

- threshold selection;
- tuning hyperparameter;
- pemilihan model terbaik;
- evaluasi tahap ini.

## 18. Interpretasi

Kesimpulan sementara Tahap 3:

1. `XGBoost` memberi `Macro F1` rata-rata terbaik pada setup utama (`0.3302`), tetapi keunggulannya tidak besar.
2. `Logistic Regression` masih kompetitif di beberapa fold dan menjadi baseline ML yang layak.
3. Baseline teknikal murni memiliki skor klasifikasi lebih rendah, tetapi `MACD + Bollinger` justru lebih baik pada hasil trading setelah biaya.
4. Pada setup sekarang, model klasifikasi cenderung overtrade dan biaya transaksi menggerus performa secara berat.

## 19. Ancaman terhadap Validitas

- threshold dipilih dari validation expanding saja; sliding belum dievaluasi;
- setup model masih konservatif dan belum mencakup kalibrasi probabilitas;
- hasil trading sangat sensitif pada fee/slippage;
- data tahun awal memiliki karakteristik likuiditas yang berbeda;
- belum ada analisis regime pasar dan final holdout.

## 20. Kesimpulan Sementara

Tahap 3 berhasil menghasilkan pipeline baseline dan model klasik yang dapat dijalankan ulang, lengkap dengan prediksi dan backtesting. Namun, hasil sementara menunjukkan bahwa:

- model tabular memang mampu mengalahkan baseline naif pada metrik klasifikasi;
- `XGBoost` menjadi kandidat utama untuk tahap berikutnya;
- performa trading model setelah fee dan slippage masih buruk;
- karena itu, penelitian ini belum boleh menyimpulkan bahwa model ML sudah profitable.

## 21. Eksperimen Tambahan: Dual-Binary BUY/SELL dengan HOLD Fallback

Eksperimen tambahan ini mengubah formulasi target dari multiclass langsung menjadi dua model biner:

- `BUY vs non-BUY`
- `SELL vs non-SELL`

Lalu aturan keputusan:

- jika hanya `BUY` aktif maka prediksi `BUY`;
- jika hanya `SELL` aktif maka prediksi `SELL`;
- jika keduanya tidak aktif maka `HOLD`;
- jika keduanya aktif maka dipilih probabilitas yang lebih tinggi.

Setup lain tetap sama:

- horizon `6 jam`
- threshold `0.50%`
- expanding window `2019-2025`
- fee `0.10%`
- slippage `0.05%`

### Perbandingan dengan multiclass

| Model dasar | Macro F1 multiclass | Macro F1 dual-binary | Delta |
|---|---:|---:|---:|
| Logistic Regression | 0.3100 | 0.2992 | -0.0108 |
| XGBoost | 0.3302 | 0.3414 | +0.0112 |

Interpretasi:

- untuk `Logistic Regression`, dual-binary justru sedikit lebih buruk;
- untuk `XGBoost`, dual-binary memberi kenaikan kecil pada `Macro F1`.

### Distribusi prediksi dual-binary test

| Model | BUY | HOLD | SELL |
|---|---:|---:|---:|
| dual_binary_logistic_regression | 20.00% | 22.90% | 57.10% |
| dual_binary_xgboost | 26.61% | 22.25% | 51.15% |

Model dual-binary tetap cenderung bias ke `SELL`, walaupun tidak seekstrem beberapa fold multiclass sebelumnya.

### Hasil backtesting dual-binary

| Model | Total Return | Sharpe | Max Drawdown | Win Rate | Trades |
|---|---:|---:|---:|---:|---:|
| dual_binary_logistic_regression | -0.9980 | -2.1694 | -0.9983 | 0.3705 | 3161 |
| dual_binary_xgboost | -0.9994 | -2.4249 | -0.9994 | 0.4187 | 3322 |

Interpretasi:

1. Formulasi dual-binary **tidak** memperbaiki hasil trading setelah biaya.
2. Bahkan ketika `XGBoost` dual-binary sedikit mengalahkan multiclass pada `Macro F1`, hasil backtest tetap sangat buruk.
3. Ini memperkuat dugaan bahwa masalah utama bukan hanya bentuk target multiclass, tetapi juga:
   - noise label;
   - kualitas edge prediksi yang kecil relatif terhadap biaya transaksi;
   - overtrading;
   - ketidakstabilan perilaku model antar-regime.

### Kesimpulan Eksperimen Dual-Binary

- Ide memisahkan `BUY` dan `SELL` layak diuji, dan sekarang sudah diuji.
- Pada data dan setup saat ini, pendekatan itu **belum** menyelesaikan masalah inti.
- Untuk tahap lanjutan, pendekatan yang lebih menjanjikan kemungkinan adalah:
  - filter confidence;
  - threshold probabilitas entry/exit;
  - pengurangan frekuensi trade;
  - regime-aware modeling;
  - atau formulasi target yang lebih selektif.

## 22. Eksperimen Tambahan: Confidence Threshold

Eksperimen ini tidak melatih model baru. Prediksi yang sudah ada difilter ulang:

- `BUY` hanya jika `probability_buy >= threshold` dan `probability_buy > probability_sell`
- `SELL` hanya jika `probability_sell >= threshold` dan `probability_sell > probability_buy`
- selain itu `HOLD`

Tujuan eksperimen ini adalah menekan overtrading dan hanya mengambil sinyal yang lebih yakin.

### Hasil seleksi threshold pada validation

| Model | Threshold terpilih | Mean validation backtest return | Mean validation Macro F1 | Mean validation trades |
|---|---:|---:|---:|---:|
| logistic_regression | 0.65 | 0.2525 | 0.2056 | 5.29 |
| xgboost | 0.50 | 0.3553 | 0.2993 | 95.14 |
| dual_binary_logistic_regression | 0.65 | 0.0942 | 0.1984 | 0.57 |
| dual_binary_xgboost | 0.50 | 0.3321 | 0.2894 | 73.00 |

Interpretasi:

- untuk `Logistic Regression`, threshold tinggi `0.65` drastis mengurangi jumlah trade;
- untuk model berbasis `XGBoost`, threshold terbaik tetap di `0.50`;
- validation menunjukkan trade-off yang sangat jelas antara `Macro F1` dan hasil trading.

### Hasil test setelah confidence filtering

| Model | Threshold | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| logistic_regression | 0.65 | 0.4486 | 0.3473 | 0.2487 |
| xgboost | 0.50 | 0.4444 | 0.3765 | 0.3453 |
| dual_binary_logistic_regression | 0.65 | 0.4447 | 0.3352 | 0.2108 |
| dual_binary_xgboost | 0.50 | 0.4418 | 0.3697 | 0.3304 |

Interpretasi:

- filtering confidence tidak otomatis menaikkan metrik klasifikasi;
- `Logistic Regression` menjadi jauh lebih konservatif dan hampir selalu `HOLD`;
- `XGBoost` paling seimbang secara klasifikasi setelah filtering.

### Hasil backtesting test setelah confidence filtering

| Model | Threshold | Total Return | Sharpe | Max Drawdown | Trades |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.65 | 32.8818 | 1.6411 | -0.4357 | 18 |
| dual_binary_logistic_regression | 0.65 | 7.3794 | 1.2471 | -0.2492 | 2 |
| dual_binary_xgboost | 0.50 | 1.0357 | 0.4534 | -0.7143 | 309 |
| xgboost | 0.50 | 0.2644 | 0.2842 | -0.7769 | 430 |

Interpretasi penting:

1. Confidence threshold adalah perubahan paling berarti sejauh ini untuk hasil trading.
2. `Logistic Regression` yang tadinya sangat buruk justru menjadi sangat baik setelah dibuat sangat selektif.
3. Peningkatan ini datang dengan trade-off besar:
   - `Macro F1` turun;
   - jumlah trade turun sangat tajam;
   - model menjadi lebih dekat ke strategi event-driven daripada klasifikasi setiap candle.
4. Ini menunjukkan bahwa masalah utama sebelumnya memang overtrading, bukan hanya akurasi mentah.

### Catatan Kehati-hatian

- Hasil ini masih berasal dari test gabungan fold pra-2026, bukan final holdout `2026 Q1`.
- Threshold confidence dipilih dari validation, jadi metodologinya masih valid untuk tahap pengembangan.
- Meski hasil trading membaik, klaim profitabilitas tetap belum boleh dibuat sebelum final holdout dan sensitivity analysis biaya.

## 23. Eksperimen Ensemble

Eksperimen ini menggabungkan probabilitas model yang sudah ada melalui `weighted soft voting`, tanpa melatih ulang base model.

Base model yang digabung:

- `logistic_regression`
- `xgboost`
- `dual_binary_logistic_regression`
- `dual_binary_xgboost`

### Seleksi Bobot di Validation

Kombinasi terbaik di validation adalah:

```text
0.5 * Logistic Regression + 0.5 * XGBoost
```

Ringkasan kandidat teratas:

| Weights | Mean Macro F1 | Mean Balanced Accuracy | Mean Validation Backtest Return |
|---|---:|---:|---:|
| `0.5 LR + 0.5 XGB` | 0.3853 | 0.4067 | -0.6345 |
| `1.0 XGB` | 0.3842 | 0.4078 | -0.6744 |
| `1.0 LR` | 0.3647 | 0.3887 | -0.6264 |

### Hasil Test Ensemble

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| XGBoost multiclass | 36.36% | 38.42% | 33.02% |
| Ensemble `0.5 LR + 0.5 XGB` | 37.69% | 39.50% | 36.96% |

Interpretasi:

- ensemble meningkatkan semua metrik klasifikasi utama dibanding `XGBoost` tunggal;
- ini adalah bukti bahwa ensemble memang berguna untuk tugas prediksi kelas;
- tetapi peningkatan klasifikasi tidak cukup besar untuk mengubah karakter dasar masalah.

### Hasil Backtesting Ensemble

| Model | Total Return | Sharpe | Max Drawdown | Trades |
|---|---:|---:|---:|---:|
| Ensemble `0.5 LR + 0.5 XGB` | -99.91% | -2.4628 | -0.9991 | 3281 |

Interpretasi:

1. Ensemble membantu klasifikasi, tetapi tidak membantu hasil trading.
2. Overtrading dan biaya transaksi masih menjadi masalah utama.
3. Untuk tujuan skripsi, ensemble tetap layak dipertahankan sebagai hasil metodologis yang sah:
   - “ensemble improves classification”
   - “ensemble does not necessarily improve trading performance”

## 24. Pilot LSTM

Karena environment awal tidak memiliki framework deep learning, implementasi `LSTM` dilakukan dengan `PyTorch`.

Untuk menjaga eksperimen tetap selesai dan stabil:

- digunakan arsitektur kecil;
- sequence length utama `24`;
- eksperimen dibatasi ke fold terbaru `2023-2025`;
- training dijalankan di CPU.

### Setup

| Komponen | Nilai |
|---|---|
| Sequence length | `24` |
| Hidden size | `32` |
| Num layers | `1` |
| Dropout | `0.2` |
| Batch size | `128` |
| Max epochs | `5` |
| Patience | `2` |
| Fold test | `2023`, `2024`, `2025` |

### Hasil klasifikasi test LSTM

Rata-rata fold test `2023-2025`:

| Metric | Nilai |
|---|---:|
| Accuracy | `43.94%` |
| Balanced Accuracy | `40.50%` |
| Macro F1 | `35.10%` |
| Weighted F1 | `40.81%` |
| MCC | `0.1346` |

### Perbandingan terhadap XGBoost pada fold terbaru

| Fold | XGBoost Accuracy | LSTM Accuracy | XGBoost Macro F1 | LSTM Macro F1 |
|---|---:|---:|---:|---:|
| 2023 | 42.55% | 47.33% | 39.47% | 39.11% |
| 2024 | 35.89% | 40.87% | 34.46% | 32.24% |
| 2025 | 35.48% | 43.62% | 32.69% | 33.94% |

Interpretasi:

- `LSTM` memberi accuracy lebih tinggi pada semua fold terbaru yang diuji;
- `Macro F1` LSTM kompetitif, tetapi tidak selalu mengalahkan `XGBoost`;
- ini menunjukkan sequence model memang menangkap sebagian konteks temporal yang tidak tertangkap model tabular.

### Hasil backtesting LSTM

| Model | Total Return | Sharpe | Max Drawdown | Trades |
|---|---:|---:|---:|---:|
| LSTM pilot | -23.24% | -0.2554 | -0.4454 | 435 |

Interpretasi:

1. `LSTM` masih belum profitable pada setup ini.
2. Namun hasil tradingnya jauh lebih baik daripada multiclass `XGBoost` mentah yang hampir habis setelah biaya.
3. Ini membuat `LSTM` layak dipertahankan sebagai pembanding deep learning dalam skripsi, walaupun belum menjadi model trading terbaik.

## File Hasil Terkait

- Metrik klasifikasi: [outputs/metrics/stage3_classification_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/stage3_classification_metrics.csv)
- Metrik backtest: [outputs/metrics/stage3_backtest_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/stage3_backtest_metrics.csv)
- Metrik dual-binary: [outputs/metrics/stage3_dual_binary_classification_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/stage3_dual_binary_classification_metrics.csv)
- Backtest dual-binary: [outputs/metrics/stage3_dual_binary_backtest_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/stage3_dual_binary_backtest_metrics.csv)
- Threshold selection: [outputs/metrics/threshold_selection_expanding.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/threshold_selection_expanding.csv)
- Prediksi model: [outputs/predictions/stage3_predictions.parquet](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/predictions/stage3_predictions.parquet)
- Prediksi dual-binary: [outputs/predictions/stage3_dual_binary_predictions.parquet](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/predictions/stage3_dual_binary_predictions.parquet)
- Seleksi confidence threshold: [outputs/metrics/stage3_confidence_threshold_selection.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/stage3_confidence_threshold_selection.csv)
- Metrik confidence threshold: [outputs/metrics/stage3_confidence_classification_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/stage3_confidence_classification_metrics.csv)
- Backtest confidence threshold: [outputs/metrics/stage3_confidence_backtest_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/stage3_confidence_backtest_metrics.csv)
- Prediksi confidence threshold: [outputs/predictions/stage3_confidence_predictions.parquet](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/predictions/stage3_confidence_predictions.parquet)
- Seleksi ensemble: [outputs/metrics/ensemble_selection.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/ensemble_selection.csv)
- Metrik ensemble: [outputs/metrics/ensemble_classification_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/ensemble_classification_metrics.csv)
- Backtest ensemble: [outputs/metrics/ensemble_backtest_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/ensemble_backtest_metrics.csv)
- Metrik LSTM: [outputs/metrics/stage4_lstm_classification_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/stage4_lstm_classification_metrics.csv)
- Backtest LSTM: [outputs/metrics/stage4_lstm_backtest_metrics.csv](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/metrics/stage4_lstm_backtest_metrics.csv)
- Confusion matrix XGBoost test: [confusion_matrix_xgboost_expanding_test.png](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/figures/confusion_matrix_xgboost_expanding_test.png)
- Confusion matrix dual-binary XGBoost test: [confusion_matrix_dual_binary_xgboost_expanding_test.png](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/outputs/figures/confusion_matrix_dual_binary_xgboost_expanding_test.png)
