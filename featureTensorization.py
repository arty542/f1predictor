import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(file_path):
    """
    Load the race data from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the race data
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None



def preprocess_data(data):
    """
    Preprocess the race data
    
    Args:
        data: DataFrame containing race data
        
    Returns:
        Processed DataFrame ready for tensorization
    """
    # Make a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Handle missing values
    numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        processed_data[col].fillna(processed_data[col].mean(), inplace=True)
    
    # Handle categorical features
    categorical_cols = ['driverId', 'constructorId', 'circuitId', 'statusId']
    for col in categorical_cols:
        if col in processed_data.columns:
            one_hot = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
            processed_data = pd.concat([processed_data, one_hot], axis=1)
            processed_data.drop(col, axis=1, inplace=True)
    
    # Feature scaling - don't scale the target variable
    target_col = 'positionOrder'
    y = processed_data[target_col]
    X = processed_data.drop(target_col, axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Scale target to 0-1 range (assuming max position is 20)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))
    
    # Combine back
    processed_data = pd.concat([X, pd.Series(y_scaled.flatten(), name=target_col)], axis=1)
    
    return processed_data, (scaler, target_scaler)

def split_data(processed_data, target_col='positionOrder', test_size=0.2, val_size=0.25):
    """
    Split data into training, validation and test sets
    
    Args:
        processed_data: Preprocessed DataFrame
        target_col: Column name of the target variable
        test_size: Proportion of data to be used as test set
        val_size: Proportion of training data to be used as validation set
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test as DataFrames
    """
    # Separate features and target
    X = processed_data.drop(target_col, axis=1)
    y = processed_data[target_col]
    
    # First split: training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Second split: training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def convert_to_tensors(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Convert pandas DataFrames to PyTorch tensors
    
    Args:
        X_train, X_val, X_test: Feature DataFrames
        y_train, y_val, y_test: Target DataFrames
        
    Returns:
        X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor
    """
    # Convert to numpy arrays first
    X_train_np = X_train.values.astype(np.float32) if hasattr(X_train, 'values') else X_train.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32) if hasattr(X_val, 'values') else X_val.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32) if hasattr(X_test, 'values') else X_test.astype(np.float32)
    
    y_train_np = y_train.values.astype(np.float32) if hasattr(y_train, 'values') else y_train.astype(np.float32)
    y_val_np = y_val.values.astype(np.float32) if hasattr(y_val, 'values') else y_val.astype(np.float32)
    y_test_np = y_test.values.astype(np.float32) if hasattr(y_test, 'values') else y_test.astype(np.float32)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    
    # For regression task, we need to reshape the target variable to be 2D
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)
    
    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

def create_data_loaders(X_train_tensor, X_val_tensor, X_test_tensor, 
                       y_train_tensor, y_val_tensor, y_test_tensor, batch_size=32):
    """
    Create PyTorch DataLoaders for training, validation and test sets
    
    Args:
        X_train_tensor, X_val_tensor, X_test_tensor: Feature tensors
        y_train_tensor, y_val_tensor, y_test_tensor: Target tensors
        batch_size: Batch size for DataLoader
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def get_input_dim(X_train):
    """
    Get input dimension for the neural network
    
    Args:
        X_train: Training features
        
    Returns:
        Number of input features
    """
    return X_train.shape[1]

def main():
    """
    Main function to run the feature tensorization pipeline
    """
    # Example usage
    file_path = 'f1dataset.csv'  # Replace with your actual data path
    
    # Load data
    data = load_data(file_path)
    
    if data is not None:
        # Preprocess data
        processed_data, scaler = preprocess_data(data)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data)
                
                # After preprocessing and splitting data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data)

        from sklearn.preprocessing import StandardScaler
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

        import pickle
        with open("checkpoints/target_scaler.pkl", "wb") as f:
            pickle.dump(target_scaler, f)

        # Use the scaled targets for tensor conversion
        X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = \
            convert_to_tensors(X_train, X_val, X_test, y_train_scaled, y_val_scaled, y_test_scaled)
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train_tensor, X_val_tensor, X_test_tensor, 
            y_train_tensor, y_val_tensor, y_test_tensor
        )
        
        # Get input dimension for the model
        input_dim = get_input_dim(X_train)
        print(f"Input dimension for the model: {input_dim}")
        
        # Return the tensors, loaders, and input dimension
        return {
            'tensors': (X_train_tensor, X_val_tensor, X_test_tensor, 
                      y_train_tensor, y_val_tensor, y_test_tensor),
            'loaders': (train_loader, val_loader, test_loader),
            'input_dim': input_dim,
            'scaler': scaler
        }
    
    return None

if __name__ == "__main__":
    main()