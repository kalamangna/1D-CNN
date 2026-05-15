import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from CSV or Excel file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    """Normalizes continuous features using StandardScaler."""
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(df[feature_cols])
    return normalized_features, scaler

def encode_labels(detection_raw: np.ndarray, class_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encodes raw string labels into binary detection values and categorical classifications.
    Expects detection labels like: 'Normal', 'Fault'.
    Expects class labels like: 'A-G', 'B-G'.
    """
    # 1. Binary Detection (0: Normal, 1: Fault)
    detection_indices = np.array([0 if str(det).strip().capitalize() == 'Normal' else 1 for det in detection_raw], dtype=np.float32)
    detection_indices = detection_indices.reshape(-1, 1)

    # 2. Multi-class Classification (One-hot encoded)
    # Clean class labels by stripping trailing spaces
    cleaned_class_raw = np.array([str(c).strip() for c in class_raw])
    
    encoder = LabelEncoder()
    integer_encoded = encoder.fit_transform(cleaned_class_raw)
    
    # We use numpy eye for one-hot encoding
    num_classes = len(encoder.classes_)
    classification_labels = np.eye(num_classes)[integer_encoded].astype(np.float32)

    return detection_indices, classification_labels, encoder

def create_windows(features: np.ndarray, detection_indices: np.ndarray, class_labels: np.ndarray, location_labels: np.ndarray,
                   window_size: int = 200, stride: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates overlapping windows for time-series data.
    Ensures windows do not cross scenario boundaries (detected by time resets in the first column).
    """
    X_list, Y_detection_list, Y_class_list, Y_location_list = [], [], [], []
    
    # Identify scenario boundaries (where t(s) resets to 0 or decreases)
    times = features[:, 0]
    boundaries = [0]
    for i in range(1, len(times)):
        if times[i] < times[i-1]:
            boundaries.append(i)
    boundaries.append(len(times))
    
    print(f"Detected {len(boundaries)-1} scenarios.")

    for b in range(len(boundaries) - 1):
        start_bound = boundaries[b]
        end_bound = boundaries[b+1]
        
        scenario_features = features[start_bound:end_bound]
        scenario_detections = detection_indices[start_bound:end_bound]
        scenario_classes = class_labels[start_bound:end_bound]
        scenario_locations = location_labels[start_bound:end_bound]
        
        if len(scenario_features) < window_size:
            continue
            
        num_windows = (len(scenario_features) - window_size) // stride + 1
        
        for i in range(num_windows):
            s_idx = i * stride
            e_idx = s_idx + window_size
            
            X_list.append(scenario_features[s_idx:e_idx])
            # Use the detection status of the last point in the window
            Y_detection_list.append(scenario_detections[e_idx - 1])
            Y_class_list.append(scenario_classes[e_idx - 1])
            Y_location_list.append(scenario_locations[e_idx - 1])

    if not X_list:
        raise ValueError("No windows could be created. Dataset might be too small or window_size too large.")

    return np.array(X_list), np.array(Y_detection_list), np.array(Y_class_list), np.array(Y_location_list)

def process_and_split_data(file_path: str, feature_cols: List[str], detection_col: str, class_col: str, location_col: str,
                           window_size: int = 200, stride: int = 50,
                           test_size: float = 0.15, val_size: float = 0.15):
    """
    Complete pipeline: Loads, normalizes, windows, encodes, and splits data.
    
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), encoder, scaler
    """
    print(f"Loading data from {file_path}...")
    df = load_data(file_path)
    
    print("Normalizing features...")
    features, scaler = normalize_features(df, feature_cols)
    detection_raw = df[detection_col].values
    class_raw = df[class_col].values
    location_raw = df[location_col].values.astype(np.float32).reshape(-1, 1)
    
    print("Encoding detection and classification...")
    detection_indices, class_labels, encoder = encode_labels(detection_raw, class_raw)
    
    print(f"Creating windows (size={window_size}, stride={stride})...")
    X, Y_detection, Y_class, Y_location = create_windows(features, detection_indices, class_labels, location_raw, window_size, stride)
    
    print("Splitting data...")
    # Calculate relative sizes
    # First split off the test set
    X_temp, X_test, Y_detection_temp, Y_detection_test, Y_class_temp, Y_class_test, Y_location_temp, Y_location_test = train_test_split(
        X, Y_detection, Y_class, Y_location, test_size=test_size, random_state=42, shuffle=True
    )
    
    # Calculate proportion of validation set relative to the remaining data
    if val_size > 0:
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, Y_detection_train, Y_detection_val, Y_class_train, Y_class_val, Y_location_train, Y_location_val = train_test_split(
            X_temp, Y_detection_temp, Y_class_temp, Y_location_temp, test_size=val_ratio, random_state=42, shuffle=True
        )
    else:
        X_train, Y_detection_train, Y_class_train, Y_location_train = X_temp, Y_detection_temp, Y_class_temp, Y_location_temp
        X_val, Y_detection_val, Y_class_val, Y_location_val = X_test, Y_detection_test, Y_class_test, Y_location_test
    
    # Package labels into dictionaries for the Keras multi-output model
    y_train = {"detection": Y_detection_train, "classification": Y_class_train, "location": Y_location_train}
    y_val = {"detection": Y_detection_val, "classification": Y_class_val, "location": Y_location_val}
    y_test = {"detection": Y_detection_test, "classification": Y_class_test, "location": Y_location_test}
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), encoder, scaler

if __name__ == "__main__":
    # --- Quick Verification Demo ---
    print("Running verification test with dummy data...")
    
    # 1. Create a dummy CSV file simulating continuous sensor recordings
    dummy_len = 2000
    dummy_data = {
        't (s)': np.linspace(0, 0.2, dummy_len),
        'Va': np.random.randn(dummy_len),
        'Vb': np.random.randn(dummy_len),
        'Vc': np.random.randn(dummy_len),
        'Ia': np.random.randn(dummy_len),
        'Ib': np.random.randn(dummy_len),
        'Ic': np.random.randn(dummy_len),
        'Detection': np.random.choice(['Normal', 'Fault'], size=dummy_len),
        'Classification': np.random.choice(['A-G Fault', 'B-G Fault'], size=dummy_len),
        'Location': np.random.rand(dummy_len)
    }
    df_dummy = pd.DataFrame(dummy_data)
    df_dummy.to_csv("dummy_dataset.csv", index=False)
    
    # 2. Process using the pipeline
    feature_columns = ['t (s)', 'Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), encoder, scaler = process_and_split_data(
            file_path="dummy_dataset.csv",
            feature_cols=feature_columns,
            detection_col='Detection',
            class_col='Classification',
            location_col='Location',
            window_size=200,
            stride=50,
            test_size=0.15,
            val_size=0.15
        )
        
        # 3. Print resulting shapes to verify they match the 1D CNN expected input
        print("\n--- Processing Complete ---")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train['detection'] shape: {y_train['detection'].shape}")
        print(f"y_train['classification'] shape: {y_train['classification'].shape}")
        print(f"y_train['location'] shape: {y_train['location'].shape}")
        
        print(f"\nX_val shape:   {X_val.shape}")
        print(f"X_test shape:  {X_test.shape}")
        print(f"\nDiscovered Classification Classes: {encoder.classes_}")
        
    finally:
        import os
        if os.path.exists("dummy_dataset.csv"):
            os.remove("dummy_dataset.csv") # Cleanup
