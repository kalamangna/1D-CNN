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
    """Plots training vs validation accuracy and loss for both heads."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss Plot
    ax1.plot(history.history['loss'], label='Total Train Loss')
    ax1.plot(history.history['val_loss'], label='Total Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy Plot (Classification head)
    ax2.plot(history.history['classification_accuracy'], label='Class Train Acc')
    ax2.plot(history.history['val_classification_accuracy'], label='Class Val Acc')
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'training_curves.png'))
    print(f"Training curves saved to {artifacts_dir}/training_curves.png")
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
    
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    FEATURE_COLS = ['t (s)', 'V(a) p.u', 'V(b) p.u', 'V(c) p.u', 'I(a) p.u', 'I(b) p.u', 'I(c) p.u']
    DETECTION_COL = 'Detection'
    CLASS_COL = 'Classification'
    LOCATION_COL = 'Fault Location (km)'
    WINDOW_SIZE = 200
    STRIDE = 50
    BATCH_SIZE = 32
    EPOCHS = 30

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
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss'),
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. Plot Learning Curves
    plot_history(history, ARTIFACTS_DIR)

    # 5. Evaluation and Metrics
    print("\nStep 4: Evaluating on Test Set...")
    # model.predict returns a list [detection_probs, class_probs, location_pred]
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

    # 7. Save Artifacts
    print("\nStep 5: Saving model and artifacts...")
    model.save(os.path.join(ARTIFACTS_DIR, 'model.keras'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'encoder.joblib'))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
    
    print(f"\nTraining complete! Artifacts saved in {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()
