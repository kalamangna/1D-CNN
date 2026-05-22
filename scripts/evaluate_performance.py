import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import json
import sys

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
FRONTEND_PUBLIC_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend', 'public', 'results')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_dataset.csv')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.keras')
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'encoder.joblib')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')

FEATURE_COLS = ['t (s)', 'V(a) p.u', 'V(b) p.u', 'V(c) p.u', 'I(a) p.u', 'I(b) p.u', 'I(c) p.u']
DETECTION_COL = 'Detection'
CLASS_COL = 'Classification'
LOCATION_COL = 'Fault Location (km)'

def main():
    if not os.path.exists(FRONTEND_PUBLIC_DIR):
        os.makedirs(FRONTEND_PUBLIC_DIR)

    # 1. Load Data
    print("Loading and preprocessing test data...")
    # Ensure we can import data_loader
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_loader import process_and_split_data
    
    _, _, (X_test, y_test_dict), encoder, _ = process_and_split_data(
        file_path=DATA_PATH,
        feature_cols=FEATURE_COLS,
        detection_col=DETECTION_COL,
        class_col=CLASS_COL,
        location_col=LOCATION_COL
    )
    
    # 2. Load Trained Model
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 3. Generate Predictions
    print("Running inference on test set...")
    predictions = model.predict(X_test)
    
    # Head 0 is 'detection' (Binary: Normal vs Fault)
    y_pred_detection_probs = predictions[0].flatten()
    y_pred_binary = (y_pred_detection_probs >= 0.5).astype(int)
    y_true_binary = y_test_dict['detection'].flatten().astype(int)
    
    # 4. Calculate Metrics
    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary)
    rec = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    # Location RMSE
    y_pred_location = predictions[2]
    y_true_location = y_test_dict['location']
    location_rmse = np.sqrt(np.mean((y_pred_location - y_true_location)**2))
    
    # Classification Metrics
    y_pred_class_probs = predictions[1]
    y_pred_class = np.argmax(y_pred_class_probs, axis=1)
    y_true_class = np.argmax(y_test_dict['classification'], axis=1)
    class_acc = accuracy_score(y_true_class, y_pred_class)

    # 5. Save Metrics to JSON
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "classification_accuracy": round(float(class_acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "location_rmse": round(float(location_rmse), 4),
        "total_samples": int(len(y_true_binary)),
        "cm": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    
    metrics_path = os.path.join(ARTIFACTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to: {metrics_path}")
    
    # 6. Confusion Matrix Visualization
    print("Generating Confusion Matrix...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fault'], 
                yticklabels=['Normal', 'Fault'])
    plt.title('Binary Confusion Matrix: Fault Detection')
    plt.xlabel('Predicted Detection')
    plt.ylabel('Actual Detection')
    
    # Save to both locations
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'binary_confusion_matrix.png'))
    plt.savefig(os.path.join(FRONTEND_PUBLIC_DIR, 'binary_confusion_matrix.png'))
    print(f"Confusion matrix saved to artifacts and frontend public.")

if __name__ == "__main__":
    main()
