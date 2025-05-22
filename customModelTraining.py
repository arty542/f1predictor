import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import os
import importlib

# Import our custom modules
from featureTensorization import load_data, preprocess_data, split_data, convert_to_tensors, create_data_loaders, get_input_dim
from customModel import create_model, RacePositionModel
from saveModelProgress import ModelSaver


class Trainer:
    """
    Class to handle the training process for our race position prediction model
    """
    def __init__(self, model, train_loader, val_loader, criterion=None, 
                optimizer=None, scheduler=None, early_stopping_patience=10,
                model_saver=None, device=None):
        """
        Initialize the trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (defaults to MSE)
            optimizer: PyTorch optimizer (defaults to Adam)
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Number of epochs to wait before early stopping
            model_saver: ModelSaver instance for saving checkpoints
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
        
        # Set data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set loss function
        if criterion is None:
            # self.criterion = nn.MSELoss()
            self.criterion = torch.nn.SmoothL1Loss()
        else:
            self.criterion = criterion
        
        # Set optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        # Set scheduler
        self.scheduler = scheduler
        
        # Set early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.epochs_no_improve = 0
        self.early_stop = False
        
        # Set model saver
        if model_saver is None:
            self.model_saver = ModelSaver()
        else:
            self.model_saver = model_saver
    
    def train_epoch(self):
        """
        Train for one epoch
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()  # Set to training mode
        running_loss = 0.0
        total_samples = 0
        
        for inputs, targets in self.train_loader:
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
        
        # Calculate average loss
        epoch_loss = running_loss / total_samples
        
        return epoch_loss
    
    def validate(self):
        """
        Validate the model on validation set
        
        Returns:
            Average validation loss
        """
        self.model.eval()  # Set to evaluation mode
        running_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():  # Turn off gradients for validation
            for inputs, targets in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate average loss
        epoch_loss = running_loss / total_samples
        
        return epoch_loss
    
    def train(self, num_epochs=100, save_freq=5):
        """
        Train the model for a specified number of epochs
        
        Args:
            num_epochs: Number of training epochs
            save_freq: Frequency to save checkpoints (every save_freq epochs)
            
        Returns:
            Final model
        """
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}/{num_epochs} - "
                 f"Train Loss: {train_loss:.4f}, "
                 f"Val Loss: {val_loss:.4f}, "
                 f"Time: {epoch_time:.2f}s")
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update model saver history
            is_best = self.model_saver.update_history(epoch, train_loss, val_loss, current_lr)
            
            # Save checkpoint periodically or if it's the best model
            if epoch % save_freq == 0 or is_best or epoch == num_epochs:
                self.model_saver.save_checkpoint(
                    self.model, self.optimizer, epoch, train_loss, val_loss, is_best
                )
            
            # Step the scheduler, if provided
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check for early stopping
            if is_best:
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    self.early_stop = True
                    break
        
        # Save history at the end of training
        self.model_saver.save_history()
        
        # Calculate total training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best model was from epoch {self.model_saver.best_epoch} "
              f"with validation loss {self.model_saver.best_val_loss:.4f}")
        
        # Plot learning curves
        self.model_saver.plot_learning_curves()
        
        # Load the best model weights
        best_model, _, _, _, _, _ = self.model_saver.load_best_model(self.model)
        
        return best_model
    
    @staticmethod
    def calculate_metrics(model, data_loader, device):
        """
        Calculate regression metrics on a dataset
        
        Args:
            model: PyTorch model
            data_loader: DataLoader for the dataset
            device: Device to run on
            
        Returns:
            Dictionary of metrics
        """
        model.eval()  # Set to evaluation mode
        
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():  # Turn off gradients
            for inputs, targets in data_loader:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Collect predictions and targets
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        # Calculate metrics
        mse = ((all_predictions - all_targets) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(all_predictions - all_targets).mean()
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }

def create_trainer(model, train_loader, val_loader, lr=0.001, weight_decay=1e-5,
                  patience=10, factor=0.5, save_dir="checkpoints"):
    """
    Create a trainer with all necessary components
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        patience: Patience for ReduceLROnPlateau
        factor: Factor by which to reduce learning rate
        save_dir: Directory to save checkpoints
        
    Returns:
        Trainer instance
    """
    # Set up loss function
    criterion = nn.MSELoss()
    
    # Set up optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience//3,
        verbose=True
    )
    
    # Set up model saver
    model_saver = ModelSaver(save_dir=save_dir)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=patience,
        model_saver=model_saver
    )
    
    return trainer
def main():
    try:
        # Load and process data
        file_path = 'f1dataset.csv'
        data = load_data(file_path)
        
        if data is None:
            print("Error: Failed to load data")
            return
            
        # Preprocess data - now returns both scalers
        processed_data, (scaler, target_scaler) = preprocess_data(data)
        
        # Save scalers
        import pickle
        with open("checkpoints/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("checkpoints/target_scaler.pkl", "wb") as f:
            pickle.dump(target_scaler, f)
            
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
        
        # Create model
        model = create_model(input_dim, hidden_sizes=[128, 64, 32])
        
        # Create trainer
        trainer = create_trainer(model, train_loader, val_loader)
        
        # Train model
        best_model = trainer.train(num_epochs=100, save_freq=5)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error in training: {e}")

if __name__ == "__main__":
    main()