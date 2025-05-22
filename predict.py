import torch
import numpy as np
import pandas as pd
import os
import pickle
import warnings

from customModel import RacePositionModel
from saveModelProgress import ModelSaver

class RacePositionPredictor:
    """
    Handles loading the trained model and making predictions on new samples.
    """
    def __init__(self, model_path, reference_columns_path=None, scaler_path=None, target_scaler_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model checkpoint and reconstruct model with correct input size
        self.model_saver = ModelSaver()
        # Load checkpoint to get architecture info
        _, _, _, _, _, model_architecture = self.model_saver.load_checkpoint(model_path)
        if model_architecture is None:
            raise ValueError("Checkpoint is missing model_architecture info.")
        input_size = model_architecture['input_size']
        hidden_sizes = model_architecture['hidden_sizes']
        output_size = model_architecture['output_size']
        self.model = RacePositionModel(input_size, hidden_sizes, output_size)
        # Now load the state dict
        self.model, _, _, _, _, _ = self.model_saver.load_checkpoint(model_path, self.model)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded and ready.")

        # Load scaler if available
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print("Warning: Scaler not found. Proceeding without scaling.")

        # Load target scaler
        self.target_scaler = None
        if target_scaler_path and os.path.exists(target_scaler_path):
            with open(target_scaler_path, "rb") as f:
                self.target_scaler = pickle.load(f)
            print(f"Target scaler loaded from {target_scaler_path}")
        else:
            raise ValueError("Target scaler is required for proper prediction scaling")

        # Load reference columns info for feature alignment
        if reference_columns_path and os.path.exists(reference_columns_path):
            with open(reference_columns_path, "rb") as f:
                self.reference_columns = pickle.load(f)
            print(f"Reference columns loaded from {reference_columns_path}")
        else:
            warnings.warn(
                "Reference columns for feature alignment not found. "
                "Trying to infer columns from sample data. "
                "Prediction may fail if columns differ from training."
            )
            self.reference_columns = None

    def preprocess_input(self, sample_df):
        """Ensures the input DataFrame matches training feature order and encoding."""
        processed = sample_df.copy()

        # --- Feature engineering: replicate exactly as in training ---
        if 'grid' in processed and 'driver_experience' in processed:
            processed['grid_per_experience'] = processed['grid'] / (processed['driver_experience'] + 1)
        if 'driver_avg_position' in processed and 'constructor_avg_position' in processed:
            processed['driver_vs_constructor_avg'] = processed['driver_avg_position'] - processed['constructor_avg_position']
        if 'driver_age' in processed:
            processed['driver_age_bin'] = pd.cut(processed['driver_age'], bins=[15, 25, 35, 50], labels=False).astype('float32')

        # One-hot encode categorical features as in training
        categorical_cols = ['driverId', 'constructorId', 'circuitId', 'statusId']
        processed = pd.get_dummies(processed, columns=[col for col in categorical_cols if col in processed.columns], drop_first=True)

        # Align columns to match training exactly, fill missing with 0 (fast, safe)
        if self.reference_columns is not None:
            processed = processed.reindex(columns=self.reference_columns, fill_value=0)
        else:
            warnings.warn("Reference columns missing; input features may not match model.")

        # Ensure all columns are numeric
        processed = processed.astype(np.float32)

        # Apply scaler if available
        if self.scaler:
            try:
                processed = pd.DataFrame(self.scaler.transform(processed), columns=processed.columns)
            except Exception as e:
                warnings.warn(f"Scaler transform failed: {e}")

        return torch.tensor(processed.values, dtype=torch.float32)

    def predict(self, sample_dict):
        """
        Accepts a dict or DataFrame with features, returns the predicted position.
        """
        if isinstance(sample_dict, dict):
            sample_df = pd.DataFrame([sample_dict])
        else:
            sample_df = sample_dict.copy()

        input_tensor = self.preprocess_input(sample_df)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Inverse transform the prediction
        pred_scaled = output.cpu().numpy().flatten()[0]
        pred = self.target_scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]
        
        return max(1, min(20, round(pred)))  # Ensure prediction is between 1-20

def load_sample_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Sample data loaded with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return None

def main():
    sample_data_path = "sample_race_entry.csv"  # Update as needed
    model_path = "checkpoints/best_model.pt"
    scaler_path = "checkpoints/scaler.pkl"
    reference_columns_path = "checkpoints/reference_columns.pkl"
    target_scaler_path = "checkpoints/target_scaler.pkl"

    sample_data = load_sample_data(sample_data_path)
    if sample_data is None:
        sample_entry = {
            'driverId': 1, 'constructorId': 1, 'circuitId': 1, 'statusId': 1,
            'grid': 5, 'driver_experience': 3, 'constructor_experience': 6,
            'driver_age': 24, 'year': 2009, 'driver_avg_position': 11.0, 'constructor_avg_position': 11.4
        }
        print("Using hardcoded sample entry")
    else:
        sample_entry = sample_data.iloc[0].to_dict()
        print("Using first row of loaded sample data")

    predictor = RacePositionPredictor(
        model_path=model_path,
        reference_columns_path=reference_columns_path,
        scaler_path=scaler_path,
        target_scaler_path=target_scaler_path
    )

    try:
        pred = predictor.predict(sample_entry)
        print("\nSample race entry:")
        for k, v in sample_entry.items():
            print(f"  {k}: {v}")
        print(f"\nPredicted race position: {pred}")
    except Exception as e:
        print(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()