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

### Penghapusan folder tidak terpakai

- Folder `notebooks/` dihapus.
- Folder `models/` dihapus.

Keduanya kosong dan tidak dipakai dalam pipeline saat ini.
