# Review Paper Referensi

Dokumen ini merangkum dua paper sebagai landasan metodologis, bukan template yang harus ditiru langsung. Fokus penelitian skripsi ini tetap pada Bitcoin, klasifikasi `BUY/HOLD/SELL`, validasi temporal, dan evaluasi trading setelah biaya.

## Paper 1

**Judul:** `Machine learning techniques for stock price prediction and graphic signal recognition`  
**Sumber file:** `papers/Machine learning techniques for stock price prediction and graphic signal recognition.pdf`

### Metode yang digunakan

- Prediksi harga menggunakan `VAR-based rolling prediction`.
- Identifikasi pola sinyal grafik menggunakan `GFNN` (`Gaussian feed-forward neural network`).
- Pembanding utama pada eksperimen prediksi harga: `TCN`, `GRU`, dan `LSTM`.

### Fitur atau indikator

- Untuk prediksi harga, paper menekankan pemodelan bersama `Open`, `High`, `Low`, `Close`.
- Untuk identifikasi sinyal grafik, paper memakai jendela data harga yang ditransposisikan.
- Sinyal grafik yang dibahas berupa pola candlestick/price chart seperti:
  - `M-shape`
  - `W-shape`
  - `Triangle`
  - `Rectangle`
  - `Wedge`
  - `Head shoulder top`
  - dan pola grafik lain yang ditag manual

### Skema pembagian data

- Data saham utama diambil sekitar `2019-01-28` sampai `2022-01-25`.
- Paper menyebut menyisihkan sebagian data sebagai test set.
- Validation set dibentuk dengan rasio acak `0.33` terhadap training.
- Ada indikasi masking pada beberapa periode terakhir untuk validasi.

### Jenis target prediksi

- Tugas 1: prediksi nilai harga OHLC masa depan.
- Tugas 2: klasifikasi label pola grafik candlestick.

### Evaluasi

- Untuk prediksi harga:
  - `Accuracy` yang didefinisikan sebagai `1 - MAPE`
  - `MSE`
  - `Explained variance`
- Untuk identifikasi sinyal grafik:
  - `micro recall`
- Paper melaporkan rata-rata accuracy prediksi sekitar `96.99%`.

### Keterbatasan

- Validation acak tidak cocok untuk time series forecasting yang ketat.
- Target utamanya bukan klasifikasi `BUY/HOLD/SELL`.
- Pengukuran `accuracy = 1 - MAPE` dapat menyesatkan bila dibaca seperti akurasi klasifikasi.
- Tidak ada pembahasan backtest trading dengan fee/slippage untuk strategi aktual.
- Pelabelan sinyal grafik bersifat manual dan sulit direproduksi dalam skripsi S1 yang fokus pada klasifikasi sinyal trading.

### Bagian yang dapat diadaptasi untuk penelitian Bitcoin

- Gagasan bahwa prediksi harga dan pengenalan sinyal adalah dua tugas berbeda: untuk skripsi ini kita akan memilih fokus yang lebih jelas, yaitu klasifikasi `BUY/HOLD/SELL`.
- Penggunaan model sequence (`LSTM`) sebagai pembanding model tabular tetap relevan.
- Ide rolling/forward prediction bisa diadaptasi menjadi `walk-forward validation`, tetapi tanpa random split.

### Bagian yang tidak diadopsi langsung

- `GFNN` untuk klasifikasi pola grafik.
- Random validation split.
- Klaim kinerja berbasis metrik yang tidak langsung terhubung dengan profit trading.

## Paper 2

**Judul:** `Predicting Buy and Sell Signals for Stocks using Bollinger Bands and MACD with the Help of Machine Learning`  
**Sumber file:** `papers/Predicting Buy and Sell Signals for Stocks using Bollinger Bands and MACD with the Help of Machine Learning.pdf`

### Metode yang digunakan

- Menggabungkan `Bollinger Bands` dan `MACD` untuk membentuk aturan sinyal teknikal.
- Menggunakan `ARIMA` untuk memprediksi tren sebagai konfirmasi sinyal.
- Strategi teknikal yang dijelaskan meliputi:
  - `Snap back to middle line`
  - `Trend reversal`
  - `M/W pattern`
  - `Bollinger squeeze`

### Fitur atau indikator

- `MACD`:
  - EMA 12
  - EMA 26
  - signal line 9
  - histogram
- `Bollinger Bands`:
  - moving average 20
  - rolling standard deviation
  - upper band
  - lower band
- OHLCV dasar.
- Paper juga menyebut `typical price` untuk komputasi Bollinger.

### Skema pembagian data

- Tiga dataset saham India (`Infosys`, `Airtel`, `TCS`) untuk strategi teknikal periode `2019-01` sampai `2020-12`.
- Dataset `Reliance` digunakan untuk ARIMA, dengan rentang lebih panjang `1996` sampai `2020`.
- Splitting ARIMA dijelaskan hanya sebagai `train-test split`, tanpa penjelasan temporal yang ketat atau walk-forward.

### Jenis target prediksi

- Bukan klasifikasi multiclass formal `BUY/HOLD/SELL` berbasis future return.
- Target praktisnya adalah sinyal buy/sell dari aturan teknikal dan arah tren hasil ARIMA.

### Evaluasi

- Klaim sekitar `98% accuracy` berasal dari `MAPE sekitar 2%` pada forecasting ARIMA.
- Tidak ada evaluasi klasifikasi terstruktur seperti `precision`, `recall`, `macro F1`, `MCC`, atau confusion matrix.
- Tidak ada backtest trading yang ketat dengan fee dan slippage.

### Keterbatasan

- Klaim akurasi tinggi tidak bisa diterjemahkan langsung sebagai kualitas sinyal trading.
- Strategi banyak berbentuk aturan heuristik visual sehingga rawan subjektivitas.
- Tidak ada final holdout temporal khusus.
- Metodologi preprocessing hilang detail; bahkan missing values disebut bisa diganti nol, yang berisiko besar.
- Tidak ada analisis leakage yang jelas.

### Bagian yang dapat diadaptasi untuk penelitian Bitcoin

- `MACD` dan `Bollinger Bands` layak dijadikan:
  - fitur model;
  - baseline teknikal;
  - pembanding terhadap model ML/DL.
- Ide konfirmasi sinyal teknikal berguna, tetapi akan diubah menjadi baseline yang terukur dan reproducible.

### Bagian yang tidak diadopsi langsung

- Penggunaan `ARIMA` sebagai model utama.
- Klaim buy/sell hanya dari visual chart tanpa label future return formal.
- Pengukuran kinerja yang tidak menghubungkan kualitas klasifikasi dengan hasil backtest setelah biaya.

## Kesimpulan Adaptasi untuk Skripsi Bitcoin

### Yang diambil

- Indikator inti `MACD` dan `Bollinger Bands`.
- Pentingnya membandingkan model tabular dan sequence model.
- Pentingnya sinyal trading yang dapat dijelaskan.

### Yang diperbaiki

- Gunakan `walk-forward validation`, bukan random split.
- Gunakan final holdout `2026 Q1`.
- Ubah target menjadi klasifikasi `BUY/HOLD/SELL` berbasis `future return`.
- Evaluasi dengan:
  - `Macro F1`
  - balanced accuracy
  - MCC
  - confusion matrix
  - distribusi probabilitas
  - backtest setelah fee dan slippage

### Posisi paper terhadap penelitian ini

- Paper 1 berguna untuk justifikasi perbandingan model ML/DL, tetapi desain validasinya tidak cukup ketat untuk langsung diikuti.
- Paper 2 berguna untuk justifikasi pemakaian indikator teknikal dan baseline rule-based, tetapi tidak cukup kuat sebagai metodologi evaluasi utama.
