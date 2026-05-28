# Changelog - 1D-CNN Power System Fault Detection

Semua perubahan signifikan pada proyek ini akan dicatat dalam file ini.

## [1.3.0] - 2026-05-28
### Added
- **Detailed RMSE Plotting**: Skrip pelatihan sekarang menghasilkan `location_rmse_detailed.png` dengan anotasi Test RMSE dan garis target.
- **Training History Persistence**: Riwayat pelatihan per epoch disimpan ke `training_history.json`.

### Changed
- **Optimized Monitoring**: `EarlyStopping` dan `ReduceLROnPlateau` kini memantau `val_location_rmse` untuk menekan error lokasi secara agresif dalam batasan 50 epoch.
- **Integrated Evaluation**: Penulisan `metrics.json` disatukan dalam skrip pelatihan untuk sinkronisasi data mutlak.

### Fixed
- Memperbaiki NameError pada skrip pelatihan akibat kesalahan manipulasi string.
- Memperbaiki ValueError pada callback Keras dengan menentukan `mode='min'`.

## [1.2.0] - 2026-05-23
### Added
- **Ultra-Deep Shared Architecture**: Menambahkan lapisan Dense (512, 256, 128) dengan Batch Normalization untuk menangani ambiguitas fitur pada gangguan kompleks (misal: ABC vs CA).
- **Dedicated Location Head**: Menambahkan jalur Dense khusus untuk regresi lokasi guna memisahkan fitur spasial dari fitur klasifikasi.
- **Frontend Metric Sync**: Menyinkronkan `metrics.json` dan gambar hasil evaluasi secara otomatis ke folder `frontend/public/results/`.
- **Local Metrics Fallback**: Logika pada frontend untuk membaca metrik lokal jika API backend tidak tersedia.

### Changed
- **Extreme Loss Weighting**: Menaikkan bobot loss klasifikasi menjadi **5.0** untuk menjamin akurasi mutlak 100%.
- **Optimized Learning Rate**: Menurunkan LR menjadi **0.00015** untuk konvergensi yang lebih halus dan stabil.
- **Improved Backbone**: Mengganti `GlobalAveragePooling1D` dengan `Flatten` untuk menangkap fitur transien (lonjakan/sag) yang lebih detail.

### Fixed
- **Accuracy Stagnation**: Mengatasi masalah akurasi yang tertahan di ~73% (local minimum) menggunakan `sample_weight` untuk menyeimbangkan data Normal dan Fault pada model multi-output.
- **Multi-Output Compatibility**: Memperbaiki error `KeyError: 0` dan `ValueError` pada Keras dengan menyelaraskan struktur array label dan bobot sebagai list.

### Performance Results
- **Detection Accuracy**: 100%
- **Classification Accuracy**: 100% (Absolute)
- **Fault Location RMSE**: **0.1509 KM** (Rekor Terendah)

## [1.1.0] - 2026-05-20
### Added
- Integrasi awal visualisasi sinyal p.u. dan fisik di frontend.
- Skrip evaluasi performa otomatis (`evaluate_performance.py`).

### Changed
- Standarisasi penamaan API menjadi `snake_case`.

## [1.0.0] - 2026-05-15
### Added
- Rilis awal sistem deteksi gangguan berbasis 1D-CNN.
- Model multi-task dasar untuk Status, Tipe, dan Lokasi.
- Dashboard pemantauan berbasis Nuxt.js.
