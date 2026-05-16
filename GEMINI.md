# 1D-CNN Power System Fault Detection

## Overview
This project implements an end-to-end system for detecting and classifying faults in electrical power systems. It leverages a 1D-Convolutional Neural Network (1D-CNN) with multi-output heads to predict status, type, and location from 3-phase voltage and current signals.

## Architecture
- **Backend:** Flask API serving a multi-task Keras model with 7-channel input.
- **Frontend:** Nuxt.js dashboard featuring deep signal visualization (p.u. and physical units).
- **Data Pipeline:** Custom processing from Excel to 7-feature windowed matrices (Time, Va, Vb, Vc, Ia, Ib, Ic).

## Project Mandates

### 1. Unified Naming Convention
- **API/JSON:** Always use **snake_case** for all response fields.
- **Keys:** `detection` (Status), `classification` (Type), `fault_location_km` (Location).
- **Simplification:** Frontend UI uses simplified terms: Verdict, Certainty, Type, Location.

### 2. Multi-Task Model Schema
- All components MUST support three simultaneous predictions:
  1. **Detection (Status):** Normal vs Fault (Sigmoid).
  2. **Classification (Type):** Specific fault scenario (Softmax).
  3. **Location:** Distance in kilometers (Snapped to discrete points: 0.25, 1.25, 2.5, 3.75, 4.9 KM).

### 3. Feature Set Consistency
- **Input Shape:** Exactly **200 time-steps** with **7 channels** (Time, Va, Vb, Vc, Ia, Ib, Ic).

### 4. Documentation Protocol
- **Pipeline:** Detailed technical documentation stored in `pipeline.pdf`.
- **User Guide:** General documentation stored in `documentation.pdf`.

## Technical Specifications (Review Findings)

### 1. Signal Processing
- **Sampling Frequency:** 10 kHz ($dt = 0.0001$).
- **Windowing Strategy:** 200 time-steps with a stride of 50.
- **Boundary Protection:** Pipeline ensures windows do not cross different fault scenarios.

### 2. Physical Constraints
- **Transmission Line:** 5.0 km (Fixed).
- **Fault Time Detection:** Based on $V_{max} < 0.85$ p.u. or $I_{max} > 0.3$ p.u. (Peak values).
- **Inception Window:** Uses 50 pre-fault samples (5ms) for context and 150 fault samples.
- **Test Lab:** Supports explicit "Normal Data" and "Fault Data" generation with calibrated 0.16 p.u. base current.

### 3. Model Performance (Current Artifacts)
- **Detection Accuracy:** 100% (Perfect Binary Classification).
- **Classification Accuracy:** 100% (Across 11 fault types).
- **Location Error:** 0.2327 KM RMSE (Measured on a 5.0 KM line).
- **Test Set:** 578 unseen samples used for final validation.

## Development Lifecycle
1. **Prepare:** Run `backend/scripts/prepare_data.py`.
2. **Train:** Run `backend/scripts/train_model.py`.
3. **Serve:** Run `backend/app.py`.
4. **Visualize:** Launch Nuxt dev server in `frontend/`.
