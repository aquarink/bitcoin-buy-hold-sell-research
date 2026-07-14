# Rencana Implementasi Rinci

## Tujuan

Membangun pipeline penelitian yang dapat diulang untuk memprediksi sinyal `BUY/HOLD/SELL` Bitcoin dari data OHLCV 1 menit yang diresample ke 1 jam, lalu membandingkan `XGBoost` dan `LSTM` melalui `walk-forward validation` pada dua skema: `expanding window` dan `sliding window`.

## Prinsip Desain

1. Semua split harus temporal.
2. Final holdout `2026 Q1` tidak boleh disentuh saat pemilihan model.
3. Feature engineering hanya memakai informasi historis sampai waktu `t`.
4. Label dibentuk dari `future_return` tetapi threshold dipilih hanya dari training-validation pra-2026.
5. Evaluasi akhir harus memisahkan:
   - kualitas klasifikasi;
   - kualitas trading setelah biaya transaksi.

## Tahap Kerja

### Tahap 1

- Audit CSV mentah secara chunked.
- Review paper referensi.
- Buat struktur proyek, konfigurasi, dan dokumentasi awal.

### Tahap 2

- Implementasi loader data mentah.
- Konversi timestamp ke `datetime UTC`.
- Resampling `1m -> 1h`:
  - `Open=first`
  - `High=max`
  - `Low=min`
  - `Close=last`
  - `Volume=sum`
- Simpan ke Parquet.
- Bangun EDA dan visualisasi utama.
- Implementasi feature engineering bebas leakage.
- Implementasi label generation untuk horizon `6h` dan `24h`.
- Implementasi splitter:
  - expanding window
  - sliding window

### Tahap 3

- Baseline:
  - always hold
  - majority class
  - MACD
  - Bollinger Bands
  - MACD + Bollinger Bands
  - Logistic Regression
- XGBoost dengan tuning temporal terbatas.
- Evaluasi klasifikasi per fold.
- Backtesting long/flat dengan `next open`, fee, dan slippage.

### Tahap 4

- LSTM sequence classification.
- Pencarian panjang sequence:
  - `24`
  - `48`
  - `72`
- Early stopping hanya memakai validation.
- Simpan history, checkpoint, dan prediksi per fold.

### Tahap 5

- Bandingkan expanding vs sliding.
- Analisis regime pasar.
- Ablation study, terutama untuk XGBoost.
- Jalankan final holdout `2026 Q1`.

### Tahap 6

- Rapikan `README`, `requirements`, tests, dan laporan akhir.
- Pastikan semua artefak tersimpan:
  - metrik
  - prediksi
  - grafik
  - model

## Keputusan Metodologis Awal

### Mengapa timeframe utama 1 jam

- Data 1 menit terlalu granular untuk eksperimen skripsi penuh.
- Timeframe 1 jam tetap cukup kaya sinyal, tetapi lebih realistis untuk:
  - XGBoost tabular
  - LSTM sequence
- walk-forward multi-fold
- backtesting berulang

### Mengapa eksperimen utama dibatasi ke 2015 ke atas

- `2012` memiliki proporsi `zero volume` yang sangat tinggi dan kualitas ekonominya paling meragukan.
- Tahun `2013-2014` lebih baik daripada `2012`, tetapi tetap mewakili fase pasar Bitcoin yang sangat awal.
- Untuk skripsi S1 yang fokus pada validitas eksperimen dan generalisasi yang lebih modern, memulai eksperimen dari `2015-01-01 UTC` lebih defensible.

### Mengapa tidak langsung memakai data 2026 untuk tuning

- Karena penelitian ini membutuhkan final holdout yang benar-benar belum terlihat.
- Tanpa pemisahan tegas, hasil final test akan bias optimistis.

### Mengapa baseline teknikal tetap dipertahankan

- Penelitian ini tidak cukup hanya menunjukkan model ML/DL punya skor klasifikasi lebih tinggi.
- Model juga harus dibandingkan dengan aturan trading sederhana yang lazim dipakai.

## Risiko yang Sudah Terlihat

1. `Zero volume` sangat tinggi pada periode awal, sehingga fitur volume harus diberi perlakuan stabil.
2. Data mulai dari `2012`, bukan `2014`, sehingga rentang awal eksperimental perlu diputuskan berdasarkan kualitas data, bukan asumsi awal.
3. `Polars` dan `DuckDB` belum tersedia di environment saat ini, jadi implementasi awal akan memakai `pandas` chunking dan `pyarrow/parquet`.

## File Tahap Berikutnya

Tahap implementasi berikutnya akan menambahkan:

- `src/config.py`
- `src/data_loader.py`
- `src/preprocessing.py`
- `src/features.py`
- `src/labels.py`
- `src/temporal_split.py`
- `tests/test_temporal_split.py`
- `tests/test_preprocessing.py`
