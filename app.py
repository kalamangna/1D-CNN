from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import os
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Config from .env
PORT = int(os.getenv('PORT', 5001))
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')

# Enable CORS for the Nuxt frontend
CORS(app, resources={r"/*": {"origins": FRONTEND_URL}})

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.keras')
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'encoder.joblib')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'metrics.json')

# Load artifacts
print("Loading model from {}...".format(MODEL_PATH))
model = tf.keras.models.load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'signals' not in data:
            return jsonify({"error": "Missing 'signals' in request body"}), 400
        
        signals = np.array(data['signals'], dtype=np.float32)
        print(f"Received signals shape: {signals.shape}")

        if signals.shape != (200, 7):
            return jsonify({"error": f"Invalid shape. Expected (200, 7), got {signals.shape}"}), 400

        # Preprocessing
        try:
            print(f"DEBUG: First row raw: {signals[0]}")
            signals_scaled = scaler.transform(signals)
            print(f"DEBUG: First row scaled: {signals_scaled[0]}")
            input_data = signals_scaled.reshape(1, 200, 7)
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return jsonify({"error": f"Scaling/Reshaping failed: {str(e)}"}), 500

        # Inference
        try:
            predictions = model.predict(input_data, verbose=0)
        except Exception as e:
            print(f"Model prediction error: {str(e)}")
            return jsonify({"error": f"Model inference failed: {str(e)}"}), 500
        
        # Multi-output extraction (Detection, Classification, Location)
        try:
            detection_prob = float(predictions[0][0][0])
            class_probs = predictions[1][0]
            detection_verdict = "Fault" if detection_prob >= 0.5 else "Normal"
            
            print(f"DEBUG: Detection Prob={detection_prob:.4f} | Verdict={detection_verdict}")
            
            predicted_idx = np.argmax(class_probs)
            confidence = float(np.max(class_probs))
            classification = encoder.classes_[predicted_idx]
            
            # Extract location
            raw_location = float(predictions[2][0][0])
            
            # Snap to nearest valid location from dataset
            # Valid: 0.25, 1.25, 2.5, 3.75, 4.9 (km)
            VALID_LOCATIONS = [0.25, 1.25, 2.5, 3.75, 4.9]
            
            if detection_verdict == "Fault":
                location_km = min(VALID_LOCATIONS, key=lambda x: abs(x - raw_location))
            else:
                location_km = 0.0
            
            # --- Detect Fault Inception Time ---
            fault_time_s = "N/A"
            if detection_verdict == "Fault":
                for i in range(len(signals)):
                    # Thresholds adjusted based on dataset analysis:
                    # Normal max I is ~0.4, Normal min V is near 0 at zero-crossings
                    # but peak V is 1.0. A sag to 0.85 or spike to 0.5 is a good indicator.
                    v_abs_max = np.max(np.abs(signals[i, 1:4]))
                    i_abs_max = np.max(np.abs(signals[i, 4:7]))
                    
                    # Detection logic: Voltage sag (peak lower than 0.85) 
                    # OR Current spike (higher than 0.3)
                    if v_abs_max < 0.85 or i_abs_max > 0.3:
                        fault_time_s = float(signals[i, 0])
                        break

            class_details = {
                encoder.classes_[i]: float(class_probs[i]) 
                for i in range(len(encoder.classes_))
            }
        except Exception as e:
            print(f"Result parsing failed. Error: {str(e)}")
            return jsonify({"error": "Model structure mismatch."}), 500

        return jsonify({
            "detection": detection_verdict,
            "classification": classification,
            "fault_location_km": round(location_km, 2),
            "fault_time_s": fault_time_s,
            "confidence": round(confidence, 4),
            "classification_detail": class_details
        })
    except Exception as e:
        print(f"Global Predict Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify({"error": "Metrics not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/artifacts/<path:filename>')
def serve_artifacts(filename):
    return send_from_directory(ARTIFACTS_DIR, filename)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"health": "ready", "model": "1D-CNN Fault Detector"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
