# Audit Dataset BTCUSD 1-Minute

## Ringkasan

- File: `data/raw/btcusd_1-min_data.csv`
- Ukuran file: `386,174,636` bytes (`368.28 MB`)
- Jumlah baris data: `7,630,675`
- Jumlah baris file termasuk header: `7,630,676`
- Kolom: `Timestamp`, `Open`, `High`, `Low`, `Close`, `Volume`
- Tipe yang layak dipakai untuk audit chunked:
  - `Timestamp`: `int64`
  - `Open`, `High`, `Low`, `Close`, `Volume`: `float32`
- Estimasi memori dense dengan dtype tersebut: sekitar `203.76 MB` di luar overhead DataFrame

## Verifikasi Rentang Waktu

- Timestamp minimum: `1325376060`
- Timestamp maksimum: `1783216500`
- Datetime minimum UTC: `2012-01-01 00:01:00+00:00`
- Datetime maksimum UTC: `2026-07-05 01:55:00+00:00`

Kesimpulan awal:

- Dataset **tidak** mulai pada 2014, tetapi mulai pada awal `2012`.
- Dataset **mencakup** seluruh `2026 Q1`.
- Dataset bahkan berlanjut melewati final holdout Q1 sampai awal `2026-07`.

## Konversi Timestamp Awal dan Akhir

Contoh baris awal:

| Timestamp | UTC | Open | High | Low | Close | Volume |
|---|---|---:|---:|---:|---:|---:|
| 1325376060 | 2012-01-01 00:01:00+00:00 | 4.58 | 4.58 | 4.58 | 4.58 | 0.0 |
| 1325376120 | 2012-01-01 00:02:00+00:00 | 4.58 | 4.58 | 4.58 | 4.58 | 0.0 |
| 1325376180 | 2012-01-01 00:03:00+00:00 | 4.58 | 4.58 | 4.58 | 4.58 | 0.0 |
| 1325376240 | 2012-01-01 00:04:00+00:00 | 4.58 | 4.58 | 4.58 | 4.58 | 0.0 |
| 1325376300 | 2012-01-01 00:05:00+00:00 | 4.58 | 4.58 | 4.58 | 4.58 | 0.0 |

Contoh baris akhir:

| Timestamp | UTC |
|---|---|
| 1783216260 | 2026-07-05 01:51:00+00:00 |
| 1783216320 | 2026-07-05 01:52:00+00:00 |
| 1783216380 | 2026-07-05 01:53:00+00:00 |
| 1783216440 | 2026-07-05 01:54:00+00:00 |
| 1783216500 | 2026-07-05 01:55:00+00:00 |

## Hasil Audit Struktur Waktu

- Missing values per kolom: semua `0`
- Timestamp duplikat: `0`
- Timestamp tidak terurut: `0`
- Missing candle 1 menit: `0`
- Interval antarbaris yang terdeteksi:

| Interval detik | Frekuensi |
|---:|---:|
| 60 | 7,630,674 |

Interpretasi:

- Dataset terlihat sebagai time series 1 menit yang **kontinu penuh** tanpa gap internal.
- Ini mengurangi kebutuhan imputasi waktu saat resampling ke 1 jam.

## Validasi OHLCV

- Nilai OHLC non-positif: `0`
- Volume negatif: `0`
- Nilai hilang: `0`

Validasi aturan OHLC:

| Aturan | Jumlah pelanggaran |
|---|---:|
| `High < Open` | 0 |
| `High < Close` | 0 |
| `High < Low` | 0 |
| `Low > Open` | 0 |
| `Low > Close` | 0 |
| `Open` di luar `[Low, High]` | 0 |
| `Close` di luar `[Low, High]` | 0 |

Kesimpulan:

- Secara struktural, OHLC konsisten.
- Tidak ada anomali harga dasar yang langsung memaksa pembersihan agresif.

## Analisis Volume Nol

- Total candle dengan `Volume = 0`: `1,312,362`

Distribusi `zero volume` per tahun:

| Tahun | Jumlah baris | Zero volume | Persentase zero volume |
|---:|---:|---:|---:|
| 2012 | 527,039 | 500,412 | 94.95% |
| 2013 | 525,600 | 205,850 | 39.17% |
| 2014 | 525,600 | 127,426 | 24.24% |
| 2015 | 525,600 | 152,865 | 29.08% |
| 2016 | 527,040 | 174,941 | 33.19% |
| 2017 | 525,600 | 42,269 | 8.04% |
| 2018 | 525,600 | 19,827 | 3.77% |
| 2019 | 525,600 | 17,261 | 3.28% |
| 2020 | 527,040 | 8,146 | 1.55% |
| 2021 | 525,600 | 3,662 | 0.70% |
| 2022 | 525,600 | 13,998 | 2.66% |
| 2023 | 525,600 | 20,735 | 3.94% |
| 2024 | 527,040 | 17,565 | 3.33% |
| 2025 | 525,600 | 5,918 | 1.13% |
| 2026* | 266,516 | 1,487 | 0.56% |

`2026*` hanya data parsial sampai `2026-07-05`.

Interpretasi awal:

- `Zero volume` sangat dominan pada tahun-tahun awal, terutama `2012`.
- Ini lebih mungkin mencerminkan karakteristik sumber data historis lama atau pasar yang sangat tipis, bukan bug sederhana.
- Untuk saat ini, baris volume nol **tidak dihapus**.
- Pada tahap berikutnya perlu dianalisis:
  - apakah zero-volume hour tetap memiliki variasi harga;
  - dampaknya pada fitur volume;
  - apakah perlu flag tambahan `is_zero_volume`.

## Jumlah Data per Tahun

| Tahun | Jumlah candle 1 menit |
|---:|---:|
| 2012 | 527,039 |
| 2013 | 525,600 |
| 2014 | 525,600 |
| 2015 | 525,600 |
| 2016 | 527,040 |
| 2017 | 525,600 |
| 2018 | 525,600 |
| 2019 | 525,600 |
| 2020 | 527,040 |
| 2021 | 525,600 |
| 2022 | 525,600 |
| 2023 | 525,600 |
| 2024 | 527,040 |
| 2025 | 525,600 |
| 2026* | 266,516 |

## Implikasi Metodologis

1. Final holdout akan tetap diambil dari `2026-01-01` sampai akhir `2026 Q1`, walaupun data tersedia lebih jauh dari itu.
2. Karena data mulai pada `2012`, eksperimen expanding/sliding dapat dimulai lebih awal daripada asumsi awal 2014.
3. Tidak adanya gap 1 menit membuat resampling 1 jam lebih aman dan reproducible.
4. Volume nol harus diperlakukan hati-hati karena dapat memengaruhi:
   - `log volume`;
   - `volume change`;
   - z-score volume;
   - strategi yang sensitif pada likuiditas.

## Pekerjaan Lanjutan Tahap Audit

- Simpan hasil audit terstruktur ke artefak mesin agar pipeline dapat dipanggil ulang.
- Audit resampling 1 jam:
  - jumlah candle per tahun;
  - deteksi candle 1 jam kosong;
  - distribusi volume nol setelah agregasi.
- Visualisasi harga, volume, return, dan volatilitas per tahun.

## Audit Resampling 1 Jam

Pipeline tahap data sekarang sudah menghasilkan `data/processed/btcusd_1h.parquet`.

### Ringkasan Resampling

- Aturan agregasi:
  - `Open = first`
  - `High = max`
  - `Low = min`
  - `Close = last`
  - `Volume = sum`
- Rentang hasil 1 jam:
  - minimum: `2012-01-01 00:00:00+00:00`
  - maksimum: `2026-07-05 01:00:00+00:00`
- Jumlah candle 1 jam: `127,178`
- Missing candle 1 jam: `0`

Interpretasi:

- Karena data 1 menit kontinu penuh, hasil 1 jam juga tidak memiliki gap internal.
- Tidak diperlukan interpolasi candle kosong pada tahap ini.

### Statistik Tahunan 1 Jam

| Tahun | Candle 1H | Zero volume 1H | Zero volume % | Rata-rata close | Rata-rata volume | Volatilitas return | Rata-rata range |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2012 | 8784 | 3283 | 37.37% | 8.24 | 64.66 | 0.0144 | 0.07 |
| 2013 | 8760 | 123 | 1.40% | 187.49 | 574.33 | 0.0200 | 4.90 |
| 2014 | 8760 | 2 | 0.02% | 525.55 | 573.54 | 0.0092 | 6.75 |
| 2015 | 8760 | 107 | 1.22% | 272.27 | 630.75 | 0.0077 | 2.63 |
| 2016 | 8784 | 0 | 0.00% | 565.84 | 226.81 | 0.0052 | 3.53 |
| 2017 | 8760 | 0 | 0.00% | 3949.13 | 536.39 | 0.0112 | 72.29 |
| 2018 | 8760 | 0 | 0.00% | 7524.25 | 448.90 | 0.0098 | 98.04 |
| 2019 | 8760 | 0 | 0.00% | 7352.62 | 341.85 | 0.0074 | 76.03 |
| 2020 | 8784 | 4 | 0.05% | 11076.14 | 349.99 | 0.0082 | 106.12 |
| 2021 | 8760 | 0 | 0.00% | 47386.70 | 205.82 | 0.0094 | 610.32 |
| 2022 | 8760 | 2 | 0.02% | 28226.67 | 95.18 | 0.0069 | 258.70 |
| 2023 | 8760 | 1 | 0.01% | 28813.02 | 77.82 | 0.0045 | 159.02 |
| 2024 | 8784 | 2 | 0.02% | 65898.07 | 91.98 | 0.0056 | 490.00 |
| 2025 | 8760 | 1 | 0.01% | 101637.87 | 73.05 | 0.0048 | 602.06 |
| 2026* | 4442 | 0 | 0.00% | 73856.98 | 99.95 | 0.0050 | 483.71 |

`2026*` masih parsial sampai `2026-07-05 01:00 UTC`.

### Implikasi Resampling

1. Timeframe utama `1H` cukup padat untuk eksperimen walk-forward lintas tahun.
2. Masalah `zero volume` sangat menurun setelah agregasi 1 jam, kecuali pada `2012`.
3. Karena `2012` masih memiliki `37.37%` hourly zero-volume candle, tahun awal perlu dipertimbangkan hati-hati saat menentukan tahun mulai eksperimen utama.

### Visualisasi yang Sudah Dibuat

- `outputs/figures/price_1h.png`
- `outputs/figures/volume_1h.png`
- `outputs/figures/return_1h.png`
- `outputs/figures/volatility_24h.png`
- `outputs/figures/hourly_candles_per_year.png`
