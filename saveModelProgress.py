import os
import torch
import json
import time
import matplotlib.pyplot as plt

class ModelSaver:
    """
    Utility class for saving and loading model checkpoints
    """
    def __init__(self, save_dir="checkpoints"):
        """
        Initialize the ModelSaver
        
        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # For tracking best model
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # For tracking training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': []
        }
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, is_best=False):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_architecture': {
                'input_size': model.input_size,
                'hidden_sizes': model.hidden_sizes,
                'output_size': model.output_size
            }
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save as best model if it's the best so far
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path, model=None, optimizer=None):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model (optional)
            optimizer: PyTorch optimizer (optional)
            
        Returns:
            model, optimizer, epoch, train_loss, val_loss, model_architecture
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return None, None, 0, 0.0, 0.0, None
        
        try:
            checkpoint = torch.load(checkpoint_path)
            
            # Load model if provided
            if model is not None:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer if provided
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            epoch = checkpoint['epoch']
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            model_architecture = checkpoint.get('model_architecture', None)
            
            print(f"Checkpoint loaded from {checkpoint_path}")
            return model, optimizer, epoch, train_loss, val_loss, model_architecture
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, None, 0, 0.0, 0.0, None
    
    def load_best_model(self, model=None, optimizer=None):
        """
        Load the best model saved so far
        
        Args:
            model: PyTorch model (optional)
            optimizer: PyTorch optimizer (optional)
            
        Returns:
            model, optimizer, epoch, train_loss, val_loss
        """
        best_path = os.path.join(self.save_dir, 'best_model.pt')
        return self.load_checkpoint(best_path, model, optimizer)
    
    def update_history(self, epoch, train_loss, val_loss, lr):
        """
        Update training history
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            lr: Current learning rate
        """
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rates'].append(lr)
        self.history['epochs'].append(epoch)
        
        # Update best model tracking
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            return True  # This is the best model so far
        return False
    
    def save_history(self):
        """
        Save training history to a JSON file
        """
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        print(f"Training history saved to {history_path}")
    
    def load_history(self):
        """
        Load training history from a JSON file
        
        Returns:
            Training history dictionary
        """
        history_path = os.path.join(self.save_dir, 'training_history.json')
        try:
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            print(f"Training history loaded from {history_path}")
            
            # Update best model tracking
            val_losses = self.history['val_loss']
            if val_losses:
                self.best_val_loss = min(val_losses)
                self.best_epoch = self.history['epochs'][val_losses.index(self.best_val_loss)]
                
            return self.history
        except Exception as e:
            print(f"Error loading training history: {e}")
            return None
    
    def plot_learning_curves(self, save_fig=True):
        """
        Plot training and validation learning curves
        
        Args:
            save_fig: Whether to save the figure
        """
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.history['epochs'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epochs'], self.history['val_loss'], label='Validation Loss')
        plt.axvline(x=self.best_epoch, color='r', linestyle='--', 
                   label=f'Best model (epoch {self.best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(1, 2, 2)
        plt.plot(self.history['epochs'], self.history['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(self.save_dir, 'learning_curves.png')
            plt.savefig(fig_path)
            print(f"Learning curves saved to {fig_path}")
        
        plt.show()

def main():
    """
    Test the ModelSaver functionality
    """
    # Just for demonstration
    model_saver = ModelSaver()
    print(f"Model saver initialized with save directory: {model_saver.save_dir}")
    
    # Simulate training history
    for epoch in range(10):
        train_loss = 1.0 / (epoch + 1)
        val_loss = 1.2 / (epoch + 1)
        lr = 0.01 * (0.9 ** epoch)
        
        is_best = model_saver.update_history(epoch, train_loss, val_loss, lr)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Best = {is_best}")
    
    # Save and plot history
    model_saver.save_history()
    model_saver.plot_learning_curves(save_fig=False)  # Don't save in this test

if __name__ == "__main__":
    main()