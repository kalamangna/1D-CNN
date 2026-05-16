---
title: 1D-CNN Fault Detection
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 1D-CNN Power System Fault Detector

An intelligent, end-to-end system for real-time detection, classification, and localization of faults in electrical transmission lines using Deep Learning.

## 🚀 Key Features
- **Multi-Task Learning:** Simultaneous Prediction of Status (Detection), Type (Classification), and Jarak (Localization mapped to 5 discrete points: 0.25, 1.25, 2.5, 3.75, 4.9 km).
- **High-Resolution Signal Analysis:** Processes 3-phase Voltage and Current signals at 10kHz.
- **Modern Dashboard:** Built with Nuxt 4, featuring real-time signal visualization in both *per-unit* and physical units.
- **Automated Pipeline:** Full lifecycle support from raw Excel data processing to model deployment.

## 📂 Architecture
- **Backend (Flask):** Serving a 1D-Convolutional Neural Network (1D-CNN) trained on multi-task objectives.
- **Frontend (Nuxt 4):** Industrial-grade UI for signal ingestion, analysis, and metric reporting.
- **AI Model:** Multi-output architecture optimized for transmission lines up to 5.0 km.

## 📂 Project Structure
- **`/backend`**: Python Flask API, Signal processing scripts, and Docker configuration.
- **`/frontend`**: Nuxt 4 application, Vercel config, and Signal visualization hub.
- **`/backend/artifacts`**: Model weights (`.keras`), Scalers, and detailed Evaluation plots (Confusion Matrices, Training Curves).

## 🐳 Deployment (Production)

### Docker Compose (Recommended)
You can deploy the entire stack using Docker:
```bash
docker-compose up -d --build
```
This will launch the backend on port `5001` and the frontend on port `3000`.

### Vercel (Frontend only)
The frontend is optimized for Vercel deployment. Ensure you set the `NUXT_PUBLIC_API_BASE_URL` environment variable to point to your deployed backend.

## 📊 Model Performance & Testing
- **Verified Metrics:** Achieved **100% Detection Accuracy**, **100% Classification Accuracy**, and a **0.23 km Location RMSE** on a 5.0 km transmission line.
- **Metrics Dashboard:** View detailed Accuracy, Precision, Recall, and F1-Score along with Training Curves and Confusion Matrices.
- **Lab (Analysis):** Test the model using explicit "Normal" or "Fault" data generation, or by uploading custom CSV data.
- **Inception Detection:** Real-time calculation of fault inception time (MS) based on voltage sag (V < 0.85) and current spike (I > 0.3) analysis.
- **Automated Tests:** Run backend logic verification using `unittest`:
  ```bash
  cd backend && venv/bin/python3 -m unittest tests/test_logic.py
  ```

## 🛠 Tech Stack
- **AI/ML:** TensorFlow 2.16+, Scikit-learn, Pandas, NumPy.
- **Backend:** Flask, Flask-CORS, Joblib.
- **Frontend:** Nuxt 4, Vue 3, Tailwind CSS, Chart.js.

## ⚡ Quick Start

### 1. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 📖 Documentation
Detailed technical documentation and user guides are available in:
- `frontend/public/docs/pipeline.pdf`
- `frontend/public/docs/documentation.pdf`
