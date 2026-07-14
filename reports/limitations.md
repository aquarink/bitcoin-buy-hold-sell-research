# Limitasi Penelitian Sementara

## Metodologi

1. Tahap ini baru menjalankan `expanding window`; `sliding window` belum dievaluasi.
2. Tahap ini baru memakai horizon utama `6 jam`; horizon `24 jam` belum diuji penuh sebagai eksperimen model.
3. Threshold label dipilih dari validation pra-2026, tetapi belum diuji lintas skema split yang lebih luas.

## Data

1. Tahun-tahun awal, terutama `2012`, memiliki proporsi volume nol yang sangat tinggi.
2. Karakteristik pasar Bitcoin berubah drastis antar-periode, sehingga satu model dapat gagal pada regime yang berbeda.

## Model

1. XGBoost dan Logistic Regression pada tahap ini masih menggunakan tuning terbatas agar komputasi tetap realistis untuk skripsi S1.
2. LSTM belum dijalankan, sehingga perbandingan ML vs DL belum lengkap.
3. Model belum memakai kalibrasi probabilitas, thresholding probabilitas, atau filtering sinyal tambahan.

## Evaluasi Trading

1. Hasil klasifikasi yang lebih baik tidak otomatis menghasilkan strategi trading yang lebih baik.
2. Fee dan slippage sangat memukul strategi dengan turnover tinggi.
3. Backtest utama masih terbatas pada long/flat dan belum mencakup sensitivity analysis biaya transaksi yang lebih luas.

## Klaim yang Tidak Boleh Dibuat Saat Ini

- Tidak boleh menyatakan model sudah profitable.
- Tidak boleh menyatakan XGBoost definitif lebih unggul dari semua pendekatan lain.
- Tidak boleh menyatakan hasil akan tetap sama pada `2026 Q1`, karena final holdout belum dievaluasi.
