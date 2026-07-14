# Catatan Perubahan Signifikan

## 2026-07-14

### Scope eksperimen dipaksa ke data 2015+

- Eksperimen model sekarang difilter eksplisit ke data `>= 2015-01-01 UTC`.
- Ini membuat seluruh runner berikutnya konsisten dengan keputusan metodologis untuk tidak memakai periode historis yang terlalu awal.

### Penambahan runner dual-binary

- Ditambahkan eksperimen `BUY vs non-BUY` dan `SELL vs non-SELL` dengan `HOLD` sebagai fallback.
- Hasilnya tidak memperbaiki performa trading secara material dibanding setup multiclass dasar.

### Penambahan runner confidence threshold

- Ditambahkan eksperimen filtering probabilitas untuk menekan overtrading.
- Ini merupakan perubahan paling signifikan sejauh ini, karena hasil trading meningkat tajam untuk beberapa model walaupun metrik klasifikasi tidak naik.

### Refactor runner target alternatif

- Runner `alt-targets` dipecah menjadi task terpisah:
  - `binary-events`
  - `direction-24h`
  - `direction-24h-neutral`
  - `regime-24h`
- Tujuannya agar eksperimen lebih cepat dan tidak perlu rerun seluruh task sekaligus.

### Hasil awal target alternatif

- `BUY vs non-BUY`, `SELL vs non-SELL`, `UP vs DOWN 24h`, dan `UP vs DOWN 24h dengan neutral filter` sudah diuji.
- Hasilnya masih belum mendekati target akurasi tinggi seperti `70%+`.
- Task yang paling stabil sejauh ini masih cenderung berada di kisaran `49%-59%` accuracy, tergantung definisi target.

### Penghapusan folder tidak terpakai

- Folder `notebooks/` dihapus.
- Folder `models/` dihapus.

Keduanya kosong dan tidak dipakai dalam pipeline saat ini.
