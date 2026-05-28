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

import json

def plot_history(history, artifacts_dir, test_rmse=None):
    """Plots training vs validation metrics and saves detailed history."""
    keys = history.history.keys()
    
    # Save raw history
    history_path = os.path.join(artifacts_dir, 'training_history.json')
    serializable_history = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)

    # ... (rest of grouping logic) ...
    loss_keys = sorted([k for k in keys if 'loss' in k and not k.startswith('val_')])
    acc_keys = sorted([k for k in keys if ('accuracy' in k or 'acc' in k) and not k.startswith('val_')])
    rmse_keys = sorted([k for k in keys if 'rmse' in k and not k.startswith('val_')])

    # --- Plot 1: Combined Overview ---
    num_plots = 1 + (1 if acc_keys else 0) + (1 if rmse_keys else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1: axes = [axes]
    
    current_ax = 0
    for k in loss_keys:
        axes[current_ax].plot(history.history[k], label=f'Train {k}')
        if f'val_{k}' in history.history:
            axes[current_ax].plot(history.history[f'val_{k}'], label=f'Val {k}', linestyle='--')
    axes[current_ax].set_title('Losses')
    axes[current_ax].legend()
    current_ax += 1

    if acc_keys:
        for k in acc_keys:
            axes[current_ax].plot(history.history[k], label=f'Train {k}')
            if f'val_{k}' in history.history:
                axes[current_ax].plot(history.history[f'val_{k}'], label=f'Val {k}', linestyle='--')
        axes[current_ax].set_title('Accuracies')
        axes[current_ax].legend()
        current_ax += 1

    if rmse_keys:
        for k in rmse_keys:
            axes[current_ax].plot(history.history[k], label=f'Train {k}')
            if f'val_{k}' in history.history:
                axes[current_ax].plot(history.history[f'val_{k}'], label=f'Val {k}', linestyle='--')
        axes[current_ax].set_title('RMSE')
        axes[current_ax].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'training_curves.png'))
    plt.close()

    # --- Plot 2: Dedicated Detailed RMSE Plot ---
    if rmse_keys:
        plt.figure(figsize=(10, 6))
        for k in rmse_keys:
            plt.plot(history.history[k], marker='o', markersize=4, alpha=0.4, label=f'Training RMSE')
            if f'val_{k}' in history.history:
                plt.plot(history.history[f'val_{k}'], marker='x', markersize=4, label=f'Validation RMSE')
        
        plt.title('Detailed Fault Location RMSE (KM) per Epoch', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('RMSE (KM)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Annotation logic: Use test_rmse if provided, otherwise last val
        final_label = "Test RMSE" if test_rmse is not None else "Final Val RMSE"
        display_val = test_rmse if test_rmse is not None else history.history[rmse_keys[0]][-1]
        
        plt.axhline(y=display_val, color='red', linestyle='--', alpha=0.3)
        plt.annotate(f'{final_label}: {display_val:.4f} KM', 
                     xy=(len(history.history[rmse_keys[0]])-1, display_val),
                     xytext=(-120, 25), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))
        
        plt.legend()
        plt.savefig(os.path.join(artifacts_dir, 'location_rmse_detailed.png'))
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
    EPOCHS = 50  # Reverted to 50 as per original successful configuration

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
    
    # Update loss weights to match the original successful configuration
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015),
        loss={
            'detection': 'binary_crossentropy',
            'classification': 'categorical_crossentropy',
            'location': 'mse'
        },
        loss_weights={
            'detection': 1.0,
            'classification': 5.0,
            'location': 0.5        # Reverted to original proven weight
        },
        metrics={
            'detection': [tf.keras.metrics.BinaryAccuracy(name='accuracy')],
            'classification': [tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
            'location': [tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        }
    )

    callbacks = [
        # Monitor val_location_rmse specifically to drive down distance error
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_location_rmse', mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, monitor='val_location_rmse', mode='min'),
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
    
    # 4. Evaluation and Metrics
    print("\nStep 4: Evaluating on Test Set...")
    predictions = model.predict(X_test)
    
    # Detection metrics
    y_pred_det_probs = predictions[0].flatten()
    y_pred_det = (y_pred_det_probs >= 0.5).astype(int)
    y_true_det = y_test['detection'].flatten().astype(int)
    
    # Classification metrics
    y_pred_probs = predictions[1] # Classification head
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test['classification'], axis=1)

    # Location evaluation
    y_pred_location = predictions[2]
    y_true_location = y_test['location']
    location_rmse = np.sqrt(np.mean((y_pred_location - y_true_location)**2))
    
    # Calculate all metrics for JSON
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_true_det, y_pred_det)
    class_acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true_det, y_pred_det)
    rec = recall_score(y_true_det, y_pred_det)
    f1 = f1_score(y_true_det, y_pred_det)
    cm = confusion_matrix(y_true_det, y_pred_det)
    tn, fp, fn, tp = cm.ravel()

    metrics_data = {
        "accuracy": round(float(acc), 4),
        "classification_accuracy": round(float(class_acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "location_rmse": round(float(location_rmse), 4),
        "total_samples": int(len(y_true_det)),
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

    # Save Metrics JSON to both locations
    for target_dir in [ARTIFACTS_DIR, FRONTEND_PUBLIC_DIR]:
        with open(os.path.join(target_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=4)
    print(f"\nFinal Test RMSE: {location_rmse:.4f} km (Saved to metrics.json)")

    # 5. Plot Learning Curves
    print("\nStep 5: Generating plots with Test Metrics...")
    plot_history(history, ARTIFACTS_DIR, test_rmse=location_rmse)
    plot_history(history, FRONTEND_PUBLIC_DIR, test_rmse=location_rmse)

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
