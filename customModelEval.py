import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import importlib

# Import our custom modules
from featureTensorization import load_data, preprocess_data, split_data, convert_to_tensors, create_data_loaders, get_input_dim
from customModel import RacePositionModel
from saveModelProgress import ModelSaver
from customModelTraining import Trainer


class ModelEvaluator:
    """
    Class to evaluate trained PyTorch models for race position prediction
    """
    def __init__(self, model, test_loader, device=None):
        """
        Initialize the evaluator
        
        Args:
            model: PyTorch model
            test_loader: Test data loader
            device: Device to run on (defaults to GPU if available, else CPU)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Set model
        self.model = model
        self.model.to(self.device)
        
        # Set test loader
        self.test_loader = test_loader
    
    def evaluate(self):
        """
        Evaluate the model on the test set
        
        Returns:
            Dictionary of metrics and arrays of targets and predictions
        """
        print("Evaluating model on test set...")
        
        self.model.eval()  # Set model to evaluation mode
        
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():  # Turn off gradients for evaluation
            for inputs, targets in self.test_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Collect predictions and targets
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())
        
        # Concatenate arrays
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        
        # Calculate metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'targets': all_targets,
            'predictions': all_predictions
        }
        
        return results
    
    def plot_predictions_vs_actual(self, targets, predictions, save_path=None):
        """
        Plot predictions vs actual values
        
        Args:
            targets: Array of actual values
            predictions: Array of predicted values
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 6))
        
        # Plot actual vs predicted
        plt.scatter(targets, predictions, alpha=0.3)
        
        # Plot perfect prediction line
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title('Predicted vs Actual Race Positions')
        plt.grid(True)
        
        # Add metrics as text
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        plt.figtext(0.15, 0.85, f"MSE: {mse:.4f}", fontsize=10)
        plt.figtext(0.15, 0.82, f"RMSE: {rmse:.4f}", fontsize=10)
        plt.figtext(0.15, 0.79, f"MAE: {mae:.4f}", fontsize=10)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_error_distribution(self, targets, predictions, save_path=None):
        """
        Plot error distribution
        
        Args:
            targets: Array of actual values
            predictions: Array of predicted values
            save_path: Path to save the plot (optional)
        """
        # Calculate errors
        errors = predictions - targets
        
        plt.figure(figsize=(12, 5))
        
        # Plot histogram of errors
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True)
        
        # Plot error vs position order
        plt.subplot(1, 2, 2)
        plt.scatter(targets, errors, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Actual Position')
        plt.ylabel('Error')
        plt.title('Error vs Actual Position')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_metrics(self, results, save_path):
        """
        Save evaluation metrics to a file
        
        Args:
            results: Dictionary of evaluation results
            save_path: Path to save the metrics
        """
        import json
        
        # Create a dict with metrics only (no numpy arrays)
        metrics = {
            'mse': float(results['mse']),
            'rmse': float(results['rmse']),
            'mae': float(results['mae']),
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {save_path}")

def load_and_evaluate_model(checkpoint_path, test_loader):
    """
    Load a model from checkpoint and evaluate it
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_loader: Test data loader
        
    Returns:
        Evaluated model and results
    """
    # Load the model
    model_saver = ModelSaver()
    model, _, _, _, _, model_architecture = model_saver.load_checkpoint(checkpoint_path)
    
    if model is None and model_architecture is not None:
        # Create model with the architecture from checkpoint
        input_size = model_architecture['input_size']
        hidden_sizes = model_architecture['hidden_sizes']
        output_size = model_architecture['output_size']
        
        model = RacePositionModel(input_size, hidden_sizes, output_size)
        
        # Try loading again
        model, _, _, _, _, _ = model_saver.load_checkpoint(checkpoint_path, model)
    
    if model is None:
        print(f"Failed to load model from {checkpoint_path}")
        return None, None
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader)
    
    # Evaluate model
    results = evaluator.evaluate()
    
    return model, results, evaluator

def main():
    """
    Main function to run evaluation
    """
    try:
        # Load and process data
        file_path = 'f1dataset.csv'
        data = load_data(file_path)
        
        if data is None:
            print("Error: Failed to load data")
            return
            
        # Preprocess data
        processed_data, scaler = preprocess_data(data)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data)
        
        # Convert to tensors
        X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = \
            convert_to_tensors(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train_tensor, X_val_tensor, X_test_tensor, 
            y_train_tensor, y_val_tensor, y_test_tensor
        )
        
        # Get input dimension for the model
        input_dim = get_input_dim(X_train)
        print(f"Input dimension for the model: {input_dim}")
        
        # Load and evaluate model
        checkpoint_path = "checkpoints/best_model.pt"
        model, results, evaluator = load_and_evaluate_model(checkpoint_path, test_loader)
        
        if model is not None and results is not None:
            # Plot results
            evaluator.plot_predictions_vs_actual(results['targets'], results['predictions'])
            evaluator.plot_error_distribution(results['targets'], results['predictions'])
            
            # Save metrics
            evaluator.save_metrics(results, "evaluation_metrics.json")
            
            print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")

if __name__ == "__main__":
    main()