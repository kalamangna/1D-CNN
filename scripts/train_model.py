import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add the backend directory to sys.path to allow imports from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import build_fault_detection_model
from scripts.data_loader import process_and_split_data

def plot_history(history, artifacts_dir):
    """Plots training vs validation metrics for all heads dynamically."""
    keys = history.history.keys()
    
    # 1. Group keys by metric type
    loss_keys = sorted([k for k in keys if 'loss' in k and not k.startswith('val_')])
    acc_keys = sorted([k for k in keys if ('accuracy' in k or 'acc' in k) and not k.startswith('val_')])
    rmse_keys = sorted([k for k in keys if 'rmse' in k and not k.startswith('val_')])

    # 2. Determine plot layout
    num_plots = 1 + (1 if acc_keys else 0) + (1 if rmse_keys else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    current_ax = 0

    # 3. Plot Losses (All in one subplot)
    for k in loss_keys:
        label = k.replace('_', ' ').title()
        axes[current_ax].plot(history.history[k], label=f'Train {label}')
        if f'val_{k}' in history.history:
            axes[current_ax].plot(history.history[f'val_{k}'], label=f'Val {label}', linestyle='--')
    axes[current_ax].set_title('Model Losses')
    axes[current_ax].set_xlabel('Epoch')
    axes[current_ax].set_ylabel('Loss Value')
    axes[current_ax].legend()
    current_ax += 1

    # 4. Plot Accuracies
    if acc_keys:
        for k in acc_keys:
            label = k.replace('_', ' ').title()
            axes[current_ax].plot(history.history[k], label=f'Train {label}')
            if f'val_{k}' in history.history:
                axes[current_ax].plot(history.history[f'val_{k}'], label=f'Val {label}', linestyle='--')
        axes[current_ax].set_title('Accuracy Metrics')
        axes[current_ax].set_xlabel('Epoch')
        axes[current_ax].set_ylabel('Accuracy (0.0 - 1.0)')
        axes[current_ax].set_ylim([-0.05, 1.05]) # Fix scale for clarity
        axes[current_ax].legend()
        current_ax += 1

    # 5. Plot RMSE (Location)
    if rmse_keys:
        for k in rmse_keys:
            label = k.replace('_', ' ').title()
            axes[current_ax].plot(history.history[k], label=f'Train {label}')
            if f'val_{k}' in history.history:
                axes[current_ax].plot(history.history[f'val_{k}'], label=f'Val {label}', linestyle='--')
        axes[current_ax].set_title('Regression Metrics (RMSE)')
        axes[current_ax].set_xlabel('Epoch')
        axes[current_ax].set_ylabel('RMSE (km)')
        axes[current_ax].legend()
        current_ax += 1

    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'training_curves.png'))
    print(f"Comprehensive training curves saved to {artifacts_dir}/training_curves.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, artifacts_dir):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Detection')
    plt.xlabel('Predicted Detection')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {artifacts_dir}/confusion_matrix.png")
    plt.close()

def main():
    # --- Configuration ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'processed_dataset.csv')
    ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
    FRONTEND_PUBLIC_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend', 'public', 'results')
    
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
    
    if not os.path.exists(FRONTEND_PUBLIC_DIR):
        os.makedirs(FRONTEND_PUBLIC_DIR)

    FEATURE_COLS = ['t (s)', 'V(a) p.u', 'V(b) p.u', 'V(c) p.u', 'I(a) p.u', 'I(b) p.u', 'I(c) p.u']
    DETECTION_COL = 'Detection'
    CLASS_COL = 'Classification'
    LOCATION_COL = 'Fault Location (km)'
    WINDOW_SIZE = 200
    STRIDE = 50
    BATCH_SIZE = 32
    EPOCHS = 50

    # 1. Load and Process Data
    print("Step 1: Processing and splitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), encoder, scaler = process_and_split_data(
        file_path=DATASET_PATH,
        feature_cols=FEATURE_COLS,
        detection_col=DETECTION_COL,
        class_col=CLASS_COL,
        location_col=LOCATION_COL,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        test_size=0.15,
        val_size=0.15
    )
    
    classes = list(encoder.classes_)
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")

    # 2. Build Model
    print("\nStep 2: Building multi-output 1D CNN model...")
    model = build_fault_detection_model(
        input_shape=(WINDOW_SIZE, len(FEATURE_COLS)),
        num_classes=num_classes
    )
    
    # 3. Training
    print("\nStep 3: Starting training...")

    # Calculate sample weights for 'detection' head to handle imbalance
    from sklearn.utils.class_weight import compute_sample_weight
    y_train_det = y_train['detection'].flatten()
    sw_detection = compute_sample_weight('balanced', y_train_det)
    
    # Convert dictionaries to lists matching the model's output order: 
    # [detection, classification, location]
    y_train_list = [y_train['detection'], y_train['classification'], y_train['location']]
    y_val_list = [y_val['detection'], y_val['classification'], y_val['location']]
    
    sample_weights_list = [
        sw_detection, 
        np.ones(len(y_train_det)), 
        np.ones(len(y_train_det))
    ]
    
    print(f"Computed Sample Weights for imbalance. Unique weights: {np.unique(sw_detection)}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, monitor='val_loss'),
    ]
    
    history = model.fit(
        X_train, y_train_list,
        validation_data=(X_val, y_val_list),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        sample_weight=sample_weights_list,
        verbose=1
    )
    
    # 4. Plot Learning Curves
    plot_history(history, ARTIFACTS_DIR)
    plot_history(history, FRONTEND_PUBLIC_DIR)

    # 5. Evaluation and Metrics
    print("\nStep 4: Evaluating on Test Set...")
    predictions = model.predict(X_test)
    y_pred_probs = predictions[1] # Classification head
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test['classification'], axis=1)

    # Location evaluation
    y_pred_location = predictions[2]
    y_true_location = y_test['location']
    location_rmse = np.sqrt(np.mean((y_pred_location - y_true_location)**2))
    print(f"\nLocation Prediction RMSE: {location_rmse:.4f} km")

    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)

    # 6. Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, classes, ARTIFACTS_DIR)
    plot_confusion_matrix(y_true, y_pred, classes, FRONTEND_PUBLIC_DIR)

    # 7. Save Artifacts
    print("\nStep 5: Saving model and artifacts...")
    model.save(os.path.join(ARTIFACTS_DIR, 'model.keras'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'encoder.joblib'))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
    
    print(f"\nTraining complete! Artifacts saved in {ARTIFACTS_DIR} and {FRONTEND_PUBLIC_DIR}")

if __name__ == "__main__":
    main()
