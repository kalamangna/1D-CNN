import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

def build_fault_detection_model(input_shape: Tuple[int, int] = (200, 7), num_classes: int = 5) -> models.Model:
    """
    Builds a 1D CNN model for multi-task fault detection in power systems.
    
    The model has three outputs:
    1. Detection: Binary classification (Normal vs Fault)
    2. Classification: Multi-class classification (Scenario)
    3. Location: Fault location in KM (Regression)
    """
    inputs = layers.Input(shape=input_shape, name="input_signals")

    # --- Shared Backbone (Feature Extraction) ---
    x = layers.Conv1D(filters=64, kernel_size=7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    shared_features = x

    # --- Output Heads ---
    # Head 1: Detection (Binary Classification - Normal vs Fault)
    detection_output = layers.Dense(1, activation='sigmoid', name='detection')(shared_features)

    # Head 2: Classification (Multi-class Classification - Scenario)
    classification_output = layers.Dense(num_classes, activation='softmax', name='classification')(shared_features)
    
    # Head 3: Location (Regression - KM)
    # Using sigmoid (0-1) then scaling by 5 to strictly bound output between 0 and 5 KM
    location_sigmoid = layers.Dense(1, activation='sigmoid', name='location_sigmoid')(shared_features)
    location_output = layers.Rescaling(scale=5.0, name='location')(location_sigmoid)

    model = models.Model(
        inputs=inputs, 
        outputs=[detection_output, classification_output, location_output], 
        name="FaultDetector_1DCNN"
    )

    # --- Compilation ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'detection': 'binary_crossentropy',
            'classification': 'categorical_crossentropy',
            'location': 'mse'
        },
        loss_weights={
            'detection': 1.0,
            'classification': 1.0,
            'location': 0.5
        },
        metrics={
            'detection': [tf.keras.metrics.BinaryAccuracy(name='accuracy')],
            'classification': [tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
            'location': [tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        }
    )

    return model

if __name__ == "__main__":
    model = build_fault_detection_model(num_classes=5)
    model.summary()
