# Backend Mandates

## Context Precedence
- Power system fault detection backend using a multi-output 1D-CNN.
- Transmission line length: **5.0 km**.
- Input: **200 time-steps** x **7 channels** (t, Va, Vb, Vc, Ia, Ib, Ic).

## Model Architecture
- **Multi-task Learning:** Three output heads:
  1. `detection` (Sigmoid): Binary classification (Normal vs Fault).
  2. `classification` (Softmax): Scenario identification based on Excel sheet names.
  3. `location` (Sigmoid * 5.0): Precise distance regression.
- **Loss Weights:** Detection (1.0), Classification (1.0), Location (0.5).

## Engineering Standards
- Use `venv/bin/python3` for execution.
- Maintain `artifacts/` as source_truth for `model.keras`, `scaler.joblib`, `encoder.joblib`.
- **API Protocol:** 
  - `POST /predict`: Input (200, 7) array. 
  - `GET /metrics`: Returns evaluation from `metrics.json`.
  - Keys: `detection`, `classification`, `fault_location_km`, `confidence`.

## Operational Workflows
- **Data Preparation:** `scripts/prepare_data.py` extracts time, signals, and calculates precise KM from sheet name percentages (e.g., 5% -> 0.25km).
- **Training:** `scripts/train_model.py` must use the 7-feature schema and multi-output backbone.
- **Scaling:** Always use `scaler.joblib` for time and signal features before inference.
