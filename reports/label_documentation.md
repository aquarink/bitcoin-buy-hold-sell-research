# Dokumentasi Label

Masalah utama penelitian ini adalah klasifikasi multiclass:

- `BUY`
- `HOLD`
- `SELL`

## Formula Dasar

Untuk horizon `h` jam:

```text
future_return_t_h = (Close[t+h] - Close[t]) / Close[t]
```

Implementasi di pipeline:

```text
future_return_h = Close.shift(-h) / Close - 1
```

## Aturan Label Fixed Threshold

- `BUY` jika `future_return > threshold`
- `SELL` jika `future_return < -threshold`
- `HOLD` jika `-threshold <= future_return <= threshold`

Jika `future_return` tidak tersedia di ujung data, label diisi `NA`.

## Horizon yang Disiapkan

- horizon utama: `6 jam`
- horizon sensitivitas: `24 jam`

## Kandidat Threshold yang Disiapkan

- `0.25%`
- `0.50%`
- `0.75%`
- `1.00%`

Threshold final **belum dipilih** pada tahap ini. Pemilihannya harus dilakukan hanya pada data training-validation pra-2026 dengan mempertimbangkan:

1. distribusi kelas;
2. Macro F1 validation;
3. hasil backtest validation setelah fee;
4. jumlah transaksi yang realistis.

## Distribusi Kelas Pra-2026

Distribusi berikut memakai seluruh data sebelum `2026-01-01` hanya untuk inspeksi awal. Ini **bukan** dasar final pemilihan threshold karena belum dipisahkan per fold validation.

### Horizon 6 Jam

| Threshold | BUY | HOLD | SELL | Catatan awal |
|---|---:|---:|---:|---|
| `0.25%` | 39.89% | 24.38% | 35.74% | HOLD relatif kecil |
| `0.50%` | 30.70% | 42.11% | 27.18% | Distribusi cukup seimbang |
| `0.75%` | 24.30% | 54.63% | 21.07% | HOLD mulai dominan |
| `1.00%` | 19.39% | 63.86% | 16.75% | HOLD sangat dominan |

### Horizon 24 Jam

| Threshold | BUY | HOLD | SELL | Catatan awal |
|---|---:|---:|---:|---|
| `0.25%` | 47.73% | 11.52% | 40.75% | HOLD terlalu kecil |
| `0.50%` | 42.56% | 21.53% | 35.91% | HOLD masih kecil |
| `0.75%` | 38.16% | 30.07% | 31.77% | Distribusi cukup seimbang |
| `1.00%` | 34.18% | 37.68% | 28.14% | Distribusi cukup seimbang |

## Implikasi Awal

1. Untuk horizon `6H`, threshold `0.50%` tampak sebagai kandidat yang masuk akal untuk diuji lebih lanjut.
2. Untuk horizon `24H`, threshold `0.75%` dan `1.00%` tampak lebih seimbang daripada threshold kecil.
3. Keputusan final tetap ditunda sampai evaluasi per fold validation dilakukan.
