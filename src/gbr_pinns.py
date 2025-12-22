#!/usr/bin/env python3
"""
GBR+PINNS: Physics-Informed Neural Networks with GBR Preprocessing

This module implements the GBR+PINNS technique that:
1. Uses GBR as a preprocessor to generate bias-corrected predictions
2. Trains a neural network constrained by the Three-Point Spectral Ansatz
3. Learns physical parameters (A, E0, E1) while fitting the data

The technique is separate from other ML methods and can be enabled/disabled independently.
"""

import numpy as np
import config

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GBR+PINNS will not work.")

# Import GBR training function (avoid circular import)
def get_gbr_trainer():
    """Get GBR trainer function to avoid circular imports."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    
    def train_gbr_model_local(X_train, y_train):
        """Local GBR training function."""
        # Use config parameters
        n_estimators = getattr(config, 'GBR_N_ESTIMATORS', 800)
        learning_rate = getattr(config, 'GBR_LEARNING_RATE', 0.02)
        max_depth = getattr(config, 'GBR_MAX_DEPTH', 3)
        min_samples_split = getattr(config, 'GBR_MIN_SAMPLES_SPLIT', 10)
        min_samples_leaf = getattr(config, 'GBR_MIN_SAMPLES_LEAF', 5)
        subsample = getattr(config, 'GBR_SUBSAMPLE', 0.8)
        n_jobs = getattr(config, 'GBR_N_JOBS', 1)
        
        gbr = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=getattr(config, 'RANDOM_SEED', 42),
            verbose=0
        )
        
        # Wrap in MultiOutputRegressor if needed
        if getattr(config, 'USE_MULTI_OUTPUT', True):
            model = MultiOutputRegressor(gbr, n_jobs=n_jobs)
        else:
            model = gbr
        
        model.fit(X_train, y_train)
        return model
    
    return train_gbr_model_local

if TORCH_AVAILABLE:
    class GBRPINNSModel(nn.Module):
        """
        Physics-Informed Neural Network for 3-point correlator prediction.
        
        The network learns both correlator predictions and physical parameters
        (Amplitude A, Ground-state Energy E0, Excited-state Energy E1).
        """
        
        def __init__(self, input_dim, output_dim, hidden_layers=None):
            super().__init__()
            
            if hidden_layers is None:
                hidden_layers = getattr(config, 'GBR_PINNS_HIDDEN_LAYERS', [64, 64, 32])
        
        activation = getattr(config, 'GBR_PINNS_ACTIVATION', 'tanh')
        
        # Build the main network for correlator prediction
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
        
        # Output layer for correlator prediction
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.correlator_network = nn.Sequential(*layers)
        
        # Learnable physical parameters (initialized with reasonable values)
        self.log_amplitude = nn.Parameter(torch.tensor(0.0))  # A parameter (log scale)
        self.log_e0 = nn.Parameter(torch.tensor(-1.0))        # E0 parameter (log scale)
        self.log_e1 = nn.Parameter(torch.tensor(-0.5))        # E1 parameter (log scale)
        
        # Store dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def get_physical_parameters(self):
        """Get the current physical parameters with positivity constraints."""
        # Use log-normal constraints to ensure positivity
        A = torch.exp(self.log_amplitude)
        E0 = torch.exp(self.log_e0)
        E1 = torch.exp(self.log_e1)
        
        # Ensure E1 > E0
        E1 = E0 + torch.exp(self.log_e1)
        
        return A, E0, E1
    
    def spectral_ansatz(self, t):
        """
        Compute the Three-Point Spectral Ansatz: C3(t) ≈ A · e^(-E0·t) · e^(-E1·(T-t)) + ...
        
        Args:
            t: Time values (tensor)
            
        Returns:
            Spectral ansatz prediction (tensor)
        """
        A, E0, E1 = self.get_physical_parameters()
        
        # For 3-point correlators, we need T (temporal extent)
        # Assume T=16 or T=25 based on input dimension
        if self.output_dim <= 16:
            T = 16
        else:
            T = 25
        
        # Three-point spectral ansatz: C3(t) = A * exp(-E0*t) * exp(-E1*(T-t))
        # Simplified to: C3(t) = A * exp(-E0*t - E1*(T-t)) = A * exp(-(E0-E1)*t - E1*T)
        spectral_pred = A * torch.exp(-E0 * t) * torch.exp(-E1 * (T - t))
        
        return spectral_pred
    
    def forward(self, x):
        """Forward pass: predict correlators using the neural network."""
        return self.correlator_network(x)
    
    def physics_loss(self, network_output, t_values):
        """
        Compute physics loss: penalize deviation from spectral ansatz.
        
        Args:
            network_output: Neural network predictions (batch_size, output_dim)
            t_values: Time values corresponding to output dimensions
            
        Returns:
            Physics loss (scalar tensor)
        """
        batch_size = network_output.shape[0]
        
        # Compute spectral ansatz for all time points
        t_tensor = torch.tensor(t_values, dtype=torch.float32, device=network_output.device)
        
        physics_loss = 0.0
        
        for i in range(batch_size):
            # Get network prediction for this sample
            net_pred = network_output[i, :]  # Shape: (output_dim,)
            
            # Compute spectral ansatz prediction
            spectral_pred = self.spectral_ansatz(t_tensor)  # Shape: (output_dim,)
            
            # Mean squared error between network output and spectral ansatz
            mse = torch.mean((net_pred - spectral_pred) ** 2)
            physics_loss += mse
        
        return physics_loss / batch_size


def prepare_gbr_pinns_data(X_train, y_train, X_bc, y_bc, gbr_model):
    """
    Step 1: Use GBR as preprocessor to generate bias-corrected data.
    
    Args:
        X_train, y_train: Training data
        X_bc, y_bc: Bias correction data  
        gbr_model: Trained GBR model
        
    Returns:
        Bias-corrected ensemble-averaged predictions for PINNS training
    """
    print("Preparing GBR+PINNS data using GBR as preprocessor...")
    
    # Get GBR predictions on bias correction data
    gbr_pred_bc = gbr_model.predict(X_bc)
    
    # Apply additive bias correction: O = (O_pred)_UD + (O_true - O_pred)_BC
    # Compute bias: bias = mean(y_bc - gbr_pred_bc)
    bias = np.mean(y_bc - gbr_pred_bc, axis=0)
    
    # Apply bias correction to training predictions
    gbr_pred_train = gbr_model.predict(X_train)
    gbr_corrected = gbr_pred_train + bias[np.newaxis, :]
    
    print(f"   Applied bias correction with bias magnitude: {np.mean(np.abs(bias)):.6f}")
    
    return gbr_corrected, y_train


def train_gbr_pinns_model(X_train, y_train, X_bc, y_bc):
    """
    Train GBR+PINNS model: GBR preprocessing + Physics-Informed Neural Network.
    
    Args:
        X_train, y_train: Training data
        X_bc, y_bc: Bias correction data
        
    Returns:
        Trained GBR+PINNS model wrapper
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GBR+PINNS but is not available.")
    
    print("Training GBR+PINNS model...")
    
    # Step 1: Train GBR model as preprocessor
    print("   Step 1: Training GBR preprocessor...")
    gbr_trainer = get_gbr_trainer()
    gbr_model = gbr_trainer(X_train, y_train)
    
    # Step 2: Prepare bias-corrected data
    gbr_corrected, y_target = prepare_gbr_pinns_data(X_train, y_train, X_bc, y_bc, gbr_model)


# Fallback function when PyTorch is not available
if not TORCH_AVAILABLE:
    def train_gbr_pinns_model(X_train, y_train, X_bc, y_bc):
        """Fallback function when PyTorch is not available."""
        raise ImportError("PyTorch is required for GBR+PINNS but is not available. Please install PyTorch or disable GBR+PINNS.")
    
    # Step 3: Setup PINNS model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = GBRPINNSModel(input_dim, output_dim)
    
    # Configuration
    lr = getattr(config, 'GBR_PINNS_LEARNING_RATE', 1e-3)
    epochs = getattr(config, 'GBR_PINNS_EPOCHS', 1000)
    batch_size = getattr(config, 'GBR_PINNS_BATCH_SIZE', 32)
    physics_weight = getattr(config, 'GBR_PINNS_PHYSICS_WEIGHT', 1.0)
    data_weight = getattr(config, 'GBR_PINNS_DATA_WEIGHT', 1.0)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(gbr_corrected, dtype=torch.float32)  # Use bias-corrected GBR data
    
    # Create data loader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Time values for physics loss
    t_values = np.arange(output_dim)
    
    print(f"   Step 2: Training PINNS (epochs={epochs}, lr={lr}, physics_weight={physics_weight})")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch_x)
            
            # Data loss (MSE between network output and GBR-corrected data)
            data_loss = torch.mean((pred - batch_y) ** 2)
            
            # Physics loss (deviation from spectral ansatz)
            physics_loss = model.physics_loss(pred, t_values)
            
            # Total loss
            total_loss = data_weight * data_loss + physics_weight * physics_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_data_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item()
            epoch_total_loss += total_loss.item()
        
        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            n_batches = len(dataloader)
            avg_data_loss = epoch_data_loss / n_batches
            avg_physics_loss = epoch_physics_loss / n_batches
            avg_total_loss = epoch_total_loss / n_batches
            
            # Get current physical parameters
            A, E0, E1 = model.get_physical_parameters()
            
            print(f"   Epoch {epoch+1:4d}: Total={avg_total_loss:.6f}, "
                  f"Data={avg_data_loss:.6f}, Physics={avg_physics_loss:.6f}, "
                  f"A={A.item():.4f}, E0={E0.item():.4f}, E1={E1.item():.4f}")
    
    # Final parameters
    A_final, E0_final, E1_final = model.get_physical_parameters()
    print(f"   Final parameters: A={A_final.item():.6f}, E0={E0_final.item():.6f}, E1={E1_final.item():.6f}")
    
    # Create wrapper for sklearn-like interface
    wrapper = GBRPINNSWrapper(gbr_model, model)
    
    return wrapper


class GBRPINNSWrapper:
    """
    Sklearn-like wrapper for GBR+PINNS model.
    """
    
    def __init__(self, gbr_model, pinns_model):
        self.gbr_model = gbr_model
        self.pinns_model = pinns_model
        
    def predict(self, X):
        """Predict using the trained GBR+PINNS model."""
        self.pinns_model.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.pinns_model(X_tensor)
            return predictions.numpy()
    
    def get_physical_parameters(self):
        """Get the learned physical parameters."""
        A, E0, E1 = self.pinns_model.get_physical_parameters()
        return {
            'amplitude': A.item(),
            'ground_state_energy': E0.item(),
            'excited_state_energy': E1.item()
        }


# Add to training registry
if hasattr(config, 'TORCH_AVAILABLE') and config.TORCH_AVAILABLE:
    # Only add if PyTorch is available
    def register_gbr_pinns():
        """Register GBR+PINNS in the training system."""
        from training import MODEL_TRAINERS
        MODEL_TRAINERS["GBR_PINNS"] = train_gbr_pinns_model
    
    # Auto-register if enabled
    if getattr(config, 'ENABLE_GBR_PINNS', False):
        try:
            register_gbr_pinns()
            print("GBR+PINNS method registered successfully")
        except Exception as e:
            print(f"Warning: Could not register GBR+PINNS: {e}")