import unittest
import numpy as np
import os
import sys
import json
import joblib
import tensorflow as tf

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from scripts.data_loader import process_and_split_data
import re

class TestFaultSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and paths."""
        cls.client = app.test_client()
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.artifacts_dir = os.path.join(cls.base_dir, 'artifacts')
        
        # Paths to artifacts
        cls.model_path = os.path.join(cls.artifacts_dir, 'model.keras')
        cls.encoder_path = os.path.join(cls.artifacts_dir, 'encoder.joblib')
        cls.scaler_path = os.path.join(cls.artifacts_dir, 'scaler.joblib')

    def test_artifacts_exist(self):
        """Ensure all required model artifacts are present."""
        self.assertTrue(os.path.exists(self.model_path), "model.keras missing")
        self.assertTrue(os.path.exists(self.encoder_path), "encoder.joblib missing")
        self.assertTrue(os.path.exists(self.scaler_path), "scaler.joblib missing")

    def test_fault_time_logic(self):
        """Tests the threshold-based fault inception time detection."""
        # Mock 200x7 signal data [t, Va, Vb, Vc, Ia, Ib, Ic]
        signals = np.zeros((200, 7))
        signals[:, 0] = np.linspace(0, 0.02, 200) # 0 to 20ms
        signals[:, 1:4] = 1.0 # Normal voltage (p.u.)
        signals[:, 4:7] = 0.5 # Normal current (p.u.)
        
        # Simulate fault at index 100 (t = 0.01s)
        # Thresholds: V < 0.8 or I > 1.2
        signals[100:, 1] = 0.7 # Phase A sag
        
        # Logic from app.py
        fault_time_s = "N/A"
        for i in range(len(signals)):
            v_abs_min = np.min(np.abs(signals[i, 1:4]))
            i_abs_max = np.max(np.abs(signals[i, 4:7]))
            if v_abs_min < 0.8 or i_abs_max > 1.2:
                fault_time_s = float(signals[i, 0])
                break
        
        self.assertNotEqual(fault_time_s, "N/A")
        self.assertAlmostEqual(fault_time_s, 0.01, places=3)

    def test_km_calculation_logic(self):
        """Tests KM extraction from scenario strings."""
        scenarios = [
            ("A-G 5%", 0.25),
            ("B-C 50%", 2.5),
            ("3PH 95%", 4.75),
            ("Normal 0%", 0.0)
        ]
        for name, expected_km in scenarios:
            km_val = 0.0
            pct_match = re.search(r'(\d+)%', name)
            if pct_match:
                pct = float(pct_match.group(1))
                km_val = (pct / 100.0) * 5.0
            self.assertEqual(km_val, expected_km)

    def test_data_splitting_logic(self):
        """Verifies that data_loader creates correct shapes and splits."""
        DATASET_PATH = os.path.join(self.base_dir, 'data', 'processed', 'processed_dataset.csv')
        if not os.path.exists(DATASET_PATH):
            self.skipTest("Processed dataset not found. Run prepare_data.py first.")
            
        FEATURE_COLS = ['t (s)', 'V(a) p.u', 'V(b) p.u', 'V(c) p.u', 'I(a) p.u', 'I(b) p.u', 'I(c) p.u']
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test), encoder, scaler = process_and_split_data(
            file_path=DATASET_PATH,
            feature_cols=FEATURE_COLS,
            detection_col='Detection',
            class_col='Classification',
            location_col='Fault Location (km)',
            test_size=0.15,
            val_size=0.15
        )
        
        # Check window shape
        self.assertEqual(X_train.shape[1:], (200, 7))
        # Check label keys
        for y in [y_train, y_val, y_test]:
            self.assertIn('detection', y)
            self.assertIn('classification', y)
            self.assertIn('location', y)

    def test_api_health(self):
        """Tests the health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['health'], 'ready')

if __name__ == '__main__':
    unittest.main()
