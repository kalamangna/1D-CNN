---
title: 1D-CNN Power System Fault Detector
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# 1D-CNN Power System Fault Detector - Backend

This is the AI-driven backend for detecting and classifying faults in power transmission lines. It uses a 1D Convolutional Neural Network (1D-CNN) with a multi-task architecture to analyze 3-phase voltage and current signals.

## Features
- **7-Channel Analysis:** Processes Time, 3x Voltage, and 3x Current signals.
- **Multi-task Learning:** Simultaneously predicts Detection (Normal/Fault), Classification (Scenario), and Location (KM).
- **Physical Bounds:** Strictly constrains distance regression within the physical line limits (0-5 KM).
- **Automated Pipeline:** Scripts for data cleaning, resampling, and multi-output training.

## Tech Stack
- **Python 3.12**
- **TensorFlow 2.16+:** Core deep learning engine.
- **Flask:** REST API framework.
- **Scikit-learn:** Data normalization and splitting.
- **FPDF2:** Automated technical documentation generator.

## Setup & Installation

1. **Environment Setup:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

2. **Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Inference:**
   ```bash
   python app.py
   ```
   The API will listen on `http://localhost:5001` (configurable via `.env`).

## Model Architecture
- **Input:** (Batch, 200, 7)
- **Shared Backbone:** 3x Conv1D layers with Batch Normalization.
- **Heads:** 
  - `detection`: Sigmoid activation.
  - `classification`: Softmax activation.
  - `location`: Rescaled Sigmoid activation (0 to 5.0).

## Data Pipeline
1. `scripts/prepare_data.py`: Processes Excel files, handles column mapping, and generates detection/classification labels from sheet names.
2. `scripts/train_model.py`: Performs training with multi-loss optimization (Binary CE, Categorical CE, and MSE).
