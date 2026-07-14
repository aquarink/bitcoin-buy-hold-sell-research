# Signal Buy Hold Sell Bitcoin

Proyek ini meneliti prediksi sinyal `BUY`, `HOLD`, dan `SELL` Bitcoin dari data OHLCV dengan perbandingan `Logistic Regression`, `XGBoost`, dan `LSTM` pada skema validasi temporal yang ketat. Fokus utama proyek adalah validitas eksperimen, pencegahan data leakage, reprodusibilitas, dan evaluasi trading setelah biaya transaksi.

## Status

Tahap aktif saat ini: `Tahap 3 - baseline, Logistic Regression, XGBoost, evaluasi klasifikasi, dan backtesting`.

Model utama belum boleh dilatih sebelum:

1. audit dataset selesai;
2. resampling dan feature engineering tervalidasi;
3. pembentukan label terdokumentasi;
4. splitter temporal dan final holdout 2026 selesai diuji.

## Struktur Proyek

```text
project/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── papers/
├── reports/
├── src/
└── tests/
```

## Cara Menjalankan

### 1. Instal dependensi

```bash
python3 -m pip install -r requirements.txt
```

### 2. Jalankan pipeline tahap data

Perintah ini akan:

- membaca CSV mentah 1 menit;
- meresample ke candle 1 jam;
- membuat fitur;
- membuat kolom label kandidat;
- menyimpan Parquet hasil;
- menyimpan statistik tahunan 1 jam;
- membuat grafik EDA awal.

```bash
python3 run_pipeline.py --stage prepare-data
```

### 3. Jalankan eksperimen klasik Tahap 3

Perintah ini akan:

- memilih threshold kandidat utama untuk horizon `6 jam` pada expanding window pra-2026;
- menjalankan baseline, `Logistic Regression`, dan `XGBoost`;
- menyimpan metrik klasifikasi;
- menyimpan prediksi per fold;
- menjalankan backtesting long/flat setelah fee dan slippage.

```bash
python3 run_pipeline.py --stage stage3-classical
```

### 4. Jalankan eksperimen dual-binary BUY/SELL dengan HOLD fallback

Perintah ini akan:

- memakai threshold utama Tahap 3 yang sama;
- melatih model biner `BUY vs non-BUY` dan `SELL vs non-SELL`;
- menggabungkan hasil dengan aturan `HOLD` fallback;
- menyimpan metrik klasifikasi, prediksi, dan backtest.

```bash
python3 run_pipeline.py --stage stage3-dual-binary
```

### 5. Jalankan eksperimen confidence threshold

Perintah ini akan:

- membaca prediksi Tahap 3 yang sudah ada;
- menguji threshold probabilitas untuk membuat model lebih selektif;
- memilih threshold berdasarkan validation;
- menyimpan metrik klasifikasi dan backtest hasil filtering.

```bash
python3 run_pipeline.py --stage stage3-confidence-threshold
```

### 6. Jalankan eksperimen target alternatif

Contoh:

```bash
python3 run_pipeline.py --stage alt-targets --task binary-events
python3 run_pipeline.py --stage alt-targets --task direction-24h
python3 run_pipeline.py --stage alt-targets --task direction-24h-neutral
python3 run_pipeline.py --stage alt-targets --task regime-24h
```

### 7. Jalankan eksperimen ensemble

```bash
python3 run_pipeline.py --stage ensemble
```

### 8. Jalankan unit test

```bash
python3 -m pytest tests
```

## Artefak yang Sudah Bisa Dihasilkan

- `data/processed/btcusd_1h.parquet`
- `data/processed/btcusd_1h_features.parquet`
- `data/processed/btcusd_1h_features_labels.parquet`
- `outputs/metrics/hourly_yearly_stats.csv`
- `outputs/metrics/threshold_selection_expanding.csv`
- `outputs/metrics/stage3_classification_metrics.csv`
- `outputs/metrics/stage3_backtest_metrics.csv`
- `outputs/metrics/stage3_dual_binary_classification_metrics.csv`
- `outputs/metrics/stage3_dual_binary_backtest_metrics.csv`
- `outputs/metrics/stage3_confidence_threshold_selection.csv`
- `outputs/metrics/stage3_confidence_classification_metrics.csv`
- `outputs/metrics/stage3_confidence_backtest_metrics.csv`
- `outputs/predictions/stage3_predictions.parquet`
- `outputs/predictions/stage3_dual_binary_predictions.parquet`
- `outputs/predictions/stage3_confidence_predictions.parquet`
- `outputs/metrics/alt_binary_event_metrics.csv`
- `outputs/metrics/alt_direction_24h_metrics.csv`
- `outputs/metrics/alt_direction_24h_neutral_metrics.csv`
- `outputs/metrics/alt_regime_metrics.csv`
- `outputs/metrics/ensemble_selection.csv`
- `outputs/metrics/ensemble_classification_metrics.csv`
- `outputs/metrics/ensemble_backtest_metrics.csv`
- `outputs/figures/confusion_matrix_xgboost_expanding_test.png`
- `outputs/figures/confusion_matrix_dual_binary_xgboost_expanding_test.png`
- `outputs/figures/price_1h.png`
- `outputs/figures/volume_1h.png`
- `outputs/figures/return_1h.png`
- `outputs/figures/volatility_24h.png`
- `outputs/figures/hourly_candles_per_year.png`

## Roadmap

1. `Tahap 1`
   Audit dataset satu menit secara efisien, verifikasi rentang tanggal aktual, review dua paper referensi, dan dokumentasi metodologi awal.
2. `Tahap 2`
   Preprocessing, resampling 1 jam, EDA, feature engineering bebas leakage, label generation, dan splitter temporal.
3. `Tahap 3`
   Baseline, Logistic Regression, XGBoost, evaluasi klasifikasi, dan backtesting long/flat.
4. `Tahap 4`
   LSTM, evaluasi sequence-based, dan perbandingan dengan model tabular.
5. `Tahap 5`
   Expanding vs sliding window, ablation, regime analysis, dan final holdout test 2026 Q1.
6. `Tahap 6`
   Rapikan pipeline, jalankan test, simpan artefak, dan tulis laporan akhir.

## Prinsip Metodologis

- Tidak ada `random train-test split`.
- Semua split berdasarkan waktu.
- Fitur hanya boleh memakai informasi sampai waktu `t`.
- Eksperimen model utama saat ini dibatasi ke data `>= 2015-01-01 UTC`.
- Data tahun 2026 untuk final holdout tidak boleh dipakai dalam pemilihan model.
- Seed utama: `42`.
- `Macro F1` menjadi metrik klasifikasi utama, bukan accuracy saja.
- Klaim trading hanya boleh dibuat setelah backtest dengan fee dan slippage.

## Lokasi Data dan Referensi

- Dataset mentah: [data/raw/btcusd_1-min_data.csv](data/raw/btcusd_1-min_data.csv)
- Paper 1: [papers/Machine learning techniques for stock price prediction and graphic signal recognition.pdf](papers/Machine%20learning%20techniques%20for%20stock%20price%20prediction%20and%20graphic%20signal%20recognition.pdf)
- Paper 2: [papers/Predicting Buy and Sell Signals for Stocks using Bollinger Bands and MACD with the Help of Machine Learning.pdf](papers/Predicting%20Buy%20and%20Sell%20Signals%20for%20Stocks%20using%20Bollinger%20Bands%20and%20MACD%20with%20the%20Help%20of%20Machine%20Learning.pdf)

## Catatan Audit Awal

- File dataset mentah berukuran sekitar `368 MB`.
- Jumlah baris awal terhitung `7,630,676` termasuk header.
- Sampel awal menunjukkan data mulai pada `2012-01-01 00:01:00 UTC`.
- Sampel akhir menunjukkan data tersedia setidaknya sampai `2026-07-05 01:55:00 UTC`, jadi final holdout 2026 Q1 tersedia dan bahkan ada data setelah Q1.
- Hasil resampling utama `1H` menghasilkan `127,178` candle tanpa missing hour.

## Dokumentasi Penting

- Audit data: [reports/data_audit.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/data_audit.md)
- Review paper: [reports/paper_review.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/paper_review.md)
- Rencana metodologi: [reports/methodology.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/methodology.md)
- Kamus fitur: [reports/feature_dictionary.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/feature_dictionary.md)
- Dokumentasi label: [reports/label_documentation.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/label_documentation.md)
- Hasil eksperimen sementara: [reports/experiment_results.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/experiment_results.md)
- Limitasi sementara: [reports/limitations.md](/Volumes/SSD%20850/PROJECTS/signal-bu-sell/reports/limitations.md)

## Batas Tahap Saat Ini

- LSTM belum dijalankan.
- Threshold label baru dipilih untuk setup utama Tahap 3, belum untuk seluruh eksperimen lanjutan.
- Final holdout `2026 Q1` belum disentuh untuk evaluasi model.
