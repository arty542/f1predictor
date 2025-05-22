import torch
import torch.nn as nn
import torch.nn.functional as F

class RacePositionModel(nn.Module):
    """
    Custom PyTorch model for race position prediction
    
    A feedforward neural network with customizable architecture:
    - Input layer → Hidden layers → Output layer
    - ReLU activations between layers
    - Dropout for regularization
    """
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], output_size=1, dropout_rate=0.3):
        """
        Initialize the model
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output layer (1 for regression)
            dropout_rate: Dropout probability for regularization
        """
        super(RacePositionModel, self).__init__()
        
        # Store the architecture parameters
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create layers dynamically based on hidden_sizes
        self.hidden_layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (predicted race position)
        """
        # Pass through hidden layers with ReLU activation and dropout after each hidden layer
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer (no activation for regression)
        x = self.output_layer(x)
        
        return x
    
    def get_num_params(self):
        """
        Calculate the number of parameters in the model
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __str__(self):
        """
        String representation of the model architecture
        
        Returns:
            String describing the model architecture
        """
        architecture = [f"Input size: {self.input_size}"]
        for i, hidden_size in enumerate(self.hidden_sizes):
            architecture.append(f"Hidden layer {i+1}: {hidden_size} neurons")
        architecture.append(f"Output size: {self.output_size}")
        architecture.append(f"Total trainable parameters: {self.get_num_params():,}")
        return "\n".join(architecture)

def create_model(input_size, hidden_sizes=[256, 128, 64, 32], output_size=1, dropout_rate=0.3):
    """
    Factory function to create a model instance
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Size of output layer (1 for regression)
        dropout_rate: Dropout probability for regularization
    
    Returns:
        Instantiated model
    """
    model = RacePositionModel(input_size, hidden_sizes, output_size, dropout_rate)
    print(model)
    return model

def main():
    """
    Test the model architecture
    """
    # Example usage
    input_size = 20  # Example input size
    model = create_model(input_size)
    
    # Test with a random input tensor
    test_input = torch.randn(10, input_size)  # Batch size of 10
    test_output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    
    return model

if __name__ == "__main__":
    main()