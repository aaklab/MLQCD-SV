#!/usr/bin/env python3
"""
Training utilities for the Lattice QCD ML pipeline.

This module isolates all machine learning model training logic:
- validating training data
- estimating MLP parameter counts
- training the Gradient Boosting Regressor (GBR)
- training the Multi-Layer Perceptron (MLP)
- training additional models (Ridge, DecisionTree)
- providing a generic train_model(...) dispatcher based on config.MODEL_TYPE

It depends only on numpy, scikit-learn and the global config.
"""

import config
import time
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


# --- NEW: PyTorch for CNN / Transformer models ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _wrap_if_multioutput(base_model, n_jobs=None):
    """
    Optionally wrap a base regressor in MultiOutputRegressor, depending on
    config.USE_MULTI_OUTPUT.

    For models that already natively support multi-output regression this is
    still safe (it will just train one model per output instead).
    """
    if getattr(config, "USE_MULTI_OUTPUT", True):
        return MultiOutputRegressor(base_model, n_jobs=n_jobs)
    return base_model


def validate_array_dimensions(array, expected_shape, array_name):
    """
    Validate array dimensions and data types.

    NOTE: This is duplicated from lattice_qcd_analysis.py so that this
    module is self-contained and avoids circular imports.
    """
    # Check if input is a numpy array
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{array_name} must be a numpy array, got {type(array)}")

    # Check for empty arrays
    if array.size == 0:
        raise ValueError(f"{array_name} must not be empty")

    # Check for numeric data type
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{array_name} must contain numeric data, got dtype {array.dtype}")

    # Check for NaN or infinite values
    if np.isnan(array).any():
        raise ValueError(f"{array_name} contains NaN values")

    if np.isinf(array).any():
        raise ValueError(f"{array_name} contains infinite values")

    # Check dimensions, if an expected shape was provided
    if expected_shape is not None:
        if len(array.shape) != len(expected_shape):
            raise ValueError(
                f"{array_name} must have {len(expected_shape)} dimensions, "
                f"got {len(array.shape)}"
            )

        for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValueError(f"{array_name} dimension {i} must be {expected}, got {actual}")

    # Check for reasonable value ranges (correlator data should not be extremely large)
    max_abs_value = np.max(np.abs(array))
    if max_abs_value > 1e10:
        raise ValueError(
            f"{array_name} contains extremely large values (max: {max_abs_value:.2e})"
        )

    # Check for extremely small values that might indicate precision issues
    # Note: Lattice QCD correlator data can legitimately have very small values
    # at large time separations.
    non_zero = array[array != 0]
    min_abs_nonzero = np.min(np.abs(non_zero)) if non_zero.size > 0 else 0.0
    if 0 < min_abs_nonzero < 1e-30:
        print(
            f"Warning: {array_name} contains very small non-zero values "
            f"(min: {min_abs_nonzero:.2e}). This may be normal for correlator "
            f"data at large time separations."
        )


def validate_model_training_inputs(X_train, y_train):
    """
    Validate inputs for machine learning model training.
    """
    # Validate array types and basic properties
    validate_array_dimensions(X_train, None, "X_train")
    validate_array_dimensions(y_train, None, "y_train")

    # Check matching number of samples
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train and y_train must have the same number of samples, "
            f"got {X_train.shape[0]} and {y_train.shape[0]}"
        )

    # Check minimum number of samples for training
    n_samples = X_train.shape[0]
    if n_samples < 10:
        raise ValueError(f"Need at least 10 samples for training, got {n_samples}")

    # Check that we have features and outputs
    if X_train.shape[1] == 0:
        raise ValueError("X_train must have at least 1 feature")

    if y_train.shape[1] == 0:
        raise ValueError("y_train must have at least 1 output")

    # Check for constant features (no variation)
    for i in range(X_train.shape[1]):
        feature = X_train[:, i]
        if np.all(feature == feature[0]):
            raise ValueError(
                f"Feature {i} in X_train has no variation (all values are {feature[0]})"
            )

    # Check for constant targets (no variation to learn)
    for i in range(y_train.shape[1]):
        target = y_train[:, i]
        if np.all(target == target[0]):
            print(
                f"Warning: Target {i} in y_train has no variation "
                f"(all values are {target[0]})"
            )

    print(
        f"Model training input validation passed: {n_samples} samples, "
        f"{X_train.shape[1]} features, {y_train.shape[1]} outputs"
    )


def estimate_mlp_parameters(hidden_layer_sizes, n_features, n_outputs):
    """
    Estimate the total number of parameters in an MLP model.
    """
    total_params = 0

    # Input to first hidden layer
    prev_size = n_features
    for layer_size in hidden_layer_sizes:
        total_params += prev_size * layer_size + layer_size  # weights + biases
        prev_size = layer_size

    # Last hidden layer to output
    total_params += prev_size * n_outputs + n_outputs  # weights + biases

    return total_params


def _get_device():
    """Return the torch device to use."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_gbr_model(X_train, y_train):
    """
    Train a Gradient Boosting Regressor model for correlator prediction.

    Uses different presets ("fast", "balanced", "accurate") controlled by
    config.GBR_MODE. All presets still use early stopping and parallelism.
    """
    validate_model_training_inputs(X_train, y_train)

    mode = getattr(config, "GBR_MODE", "fast").lower()

    # ---- Presets ----
    if mode == "fast":
        default_params = dict(
            n_estimators=30,
            learning_rate=0.15,
            max_depth=2,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            n_iter_no_change=5,
            validation_fraction=0.1,
        )
    elif mode == "balanced":
        default_params = dict(
            n_estimators=60,      # more stages than "fast"
            learning_rate=0.10,   # a bit smaller step
            max_depth=3,          # more expressive than depth=2
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            n_iter_no_change=5,
            validation_fraction=0.1,
        )
    elif mode == "accurate":
        default_params = dict(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=3,
            min_samples_leaf=1,
            subsample=0.9,
            n_iter_no_change=8,
            validation_fraction=0.1,
        )
    else:
        raise ValueError(f"Unknown GBR_MODE '{mode}'")

    # Allow overrides from config.py, falling back to the preset values
    n_estimators        = getattr(config, "GBR_N_ESTIMATORS",        default_params["n_estimators"])
    learning_rate       = getattr(config, "GBR_LEARNING_RATE",       default_params["learning_rate"])
    max_depth           = getattr(config, "GBR_MAX_DEPTH",           default_params["max_depth"])
    min_samples_split   = getattr(config, "GBR_MIN_SAMPLES_SPLIT",   default_params["min_samples_split"])
    min_samples_leaf    = getattr(config, "GBR_MIN_SAMPLES_LEAF",    default_params["min_samples_leaf"])
    subsample           = getattr(config, "GBR_SUBSAMPLE",           default_params["subsample"])
    n_iter_no_change    = getattr(config, "GBR_N_ITER_NO_CHANGE",    default_params["n_iter_no_change"])
    validation_fraction = getattr(config, "GBR_VALIDATION_FRACTION", default_params["validation_fraction"])
    n_jobs              = getattr(config, "GBR_N_JOBS",              -1)  # -1 = all cores
    loss                = getattr(config, "GBR_LOSS",                "squared_error")

    gbr_base = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=config.RANDOM_SEED,
        loss=loss,
        n_iter_no_change=n_iter_no_change,
        validation_fraction=validation_fraction,
    )

    gbr_model = _wrap_if_multioutput(gbr_base, n_jobs=n_jobs)

    print(
        f"Training GBR model (mode='{mode}') on {X_train.shape[0]} samples "
        f"with {X_train.shape[1]} features..."
    )
    print(
        f"GBR config: n_estimators={n_estimators}, max_depth={max_depth}, "
        f"learning_rate={learning_rate}, subsample={subsample}, "
        f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
        f"n_iter_no_change={n_iter_no_change}, n_jobs={n_jobs}, loss='{loss}'"
    )
    print(f"Target shape: {y_train.shape} (predicting {y_train.shape[1]} time slices)")

    gbr_model.fit(X_train, y_train)

    print("GBR model training completed successfully")

    return gbr_model


def train_ridge_model(X_train, y_train):
    """
    Train a Ridge regression model for correlator prediction.
    """
    validate_model_training_inputs(X_train, y_train)

    alpha = getattr(config, "RIDGE_ALPHA", 1.0)
    ridge_base = Ridge(alpha=alpha)

    model = _wrap_if_multioutput(ridge_base)

    print(
        f"Training Ridge model on {X_train.shape[0]} samples "
        f"with {X_train.shape[1]} features... (alpha={alpha})"
    )
    print(f"Target shape: {y_train.shape} (predicting {y_train.shape[1]} time slices)")

    model.fit(X_train, y_train)

    print("Ridge model training completed successfully")

    return model


def train_dtree_model(X_train, y_train):
    """
    Train a Decision Tree Regressor model for correlator prediction.
    """
    validate_model_training_inputs(X_train, y_train)

    max_depth = getattr(config, "DTREE_MAX_DEPTH", None)
    min_samples_leaf = getattr(config, "DTREE_MIN_SAMPLES_LEAF", 1)

    dtree_base = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=config.RANDOM_SEED,
    )

    model = _wrap_if_multioutput(dtree_base)

    print(
        f"Training DecisionTree model on {X_train.shape[0]} samples "
        f"with {X_train.shape[1]} features..."
    )
    print(
        f"DecisionTree config: max_depth={max_depth}, "
        f"min_samples_leaf={min_samples_leaf}"
    )
    print(f"Target shape: {y_train.shape} (predicting {y_train.shape[1]} time slices)")

    model.fit(X_train, y_train)

    print("DecisionTree model training completed successfully")

    return model


def train_mlp_model(X_train, y_train):
    """
    Train a Multi-Layer Perceptron model for correlator prediction with improved
    configuration to prevent model collapse. Uses a single MLPRegressor with a
    Pipeline for proper input scaling.
    """
    # Validate model training inputs
    validate_model_training_inputs(X_train, y_train)

    # Configure MLPRegressor with improved hyperparameters
    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
        activation=config.MLP_ACTIVATION,
        solver=config.MLP_SOLVER,
        alpha=config.MLP_ALPHA,
        learning_rate=config.MLP_LEARNING_RATE,
        learning_rate_init=config.MLP_LR_INIT,
        max_iter=config.MLP_MAX_ITER,
        tol=config.MLP_TOL,
        random_state=config.MLP_RANDOM_STATE,
        early_stopping=config.MLP_EARLY_STOPPING,
        validation_fraction=config.MLP_VALIDATION_FRACTION,
        n_iter_no_change=config.MLP_N_ITER_NO_CHANGE,
        shuffle=config.MLP_SHUFFLE,
        batch_size=config.MLP_BATCH_SIZE,
        beta_1=config.MLP_BETA_1,
        beta_2=config.MLP_BETA_2,
        epsilon=config.MLP_EPSILON,
        verbose=config.MLP_VERBOSE,
    )

    # Create Pipeline with StandardScaler for input preprocessing
    mlp_model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp_regressor),
    ])

    # Report architecture / parameter count
    n_features = X_train.shape[1]
    n_outputs = y_train.shape[1]
    hidden_layers = config.MLP_HIDDEN_LAYERS
    total_params = estimate_mlp_parameters(hidden_layers, n_features, n_outputs)

    print(
        f"Training MLP model (single regressor) on {X_train.shape[0]} samples "
        f"with {n_features} features..."
    )
    print(f"Target shape: {y_train.shape} (predicting {n_outputs} time slices)")
    print(f"Architecture: {hidden_layers}, total parameters ≈ {total_params:,}")
    print(
        f"Max iterations: {mlp_regressor.max_iter}, "
        f"early-stopping patience: {mlp_regressor.n_iter_no_change}"
    )
    print(f"Batch size: {mlp_regressor.batch_size}")
    print("IMPORTANT: Using single MLPRegressor (not MultiOutputRegressor) to prevent model collapse")
    print("This may take several minutes due to the large architecture...")
    print("Progress will be shown below (verbose output from sklearn):")
    print("-" * 60)

    start_time = time.time()
    mlp_model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print("-" * 60)
    print(
        f"MLP model training completed successfully in {training_time:.1f} seconds "
        f"({training_time/60:.1f} minutes)"
    )

    # Report final training statistics
    mlp_regressor_trained = mlp_model.named_steps["mlp"]
    if hasattr(mlp_regressor_trained, "n_iter_"):
        print(f"Training iterations: {mlp_regressor_trained.n_iter_}")

    if hasattr(mlp_regressor_trained, "loss_"):
        print(f"Final training loss: {mlp_regressor_trained.loss_:.6f}")

    if hasattr(mlp_regressor_trained, "validation_scores_"):
        final_val_score = mlp_regressor_trained.validation_scores_[-1]
        print(f"Final validation score: {final_val_score:.6f}")

    return mlp_model

# ---------------------------------------------------------------------------
# Torch helpers and additional models (Ridge / Tree / CNN / Transformer)
# ---------------------------------------------------------------------------


def _get_device():
    """Return the torch device to use."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _TorchRegressorBase:
    """
    Simple sklearn-like wrapper for torch models:
    - fit(X, y) with numpy arrays
    - predict(X) returns numpy arrays
    """

    def __init__(self, model, n_epochs=50, lr=1e-3, batch_size=64):
        self.model = model
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = _get_device()

    def fit(self, X, y):
        self.model.to(self.device)
        self.model.train()

        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y, dtype=np.float32))

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        t0 = time.time()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                preds = self.model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            # Only print a few epochs to avoid spam
            if epoch < 5 or (epoch + 1) % 10 == 0 or epoch == self.n_epochs - 1:
                print(f"   [epoch {epoch+1:3d}/{self.n_epochs}] loss = {epoch_loss:.4e}")

        dt = time.time() - t0
        print(f"   Torch model training completed in {dt:.2f} s")

    def predict(self, X):
        self.model.to(self.device)
        self.model.eval()

        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                out = self.model(xb)
                preds.append(out.cpu().numpy())

        return np.vstack(preds)


# ---------------------------------------------------------------------------
# Ridge / Decision Tree models (pure sklearn)
# ---------------------------------------------------------------------------


def train_ridge_model(X_train, y_train):
    """
    Train a Ridge Regression model for correlator prediction.
    """
    validate_model_training_inputs(X_train, y_train)

    alpha = getattr(config, "RIDGE_ALPHA", 1.0)
    random_state = getattr(config, "RANDOM_SEED", 123)

    base_regressor = Ridge(alpha=alpha, random_state=random_state)

    if getattr(config, "USE_MULTI_OUTPUT", True):
        print("Wrapping Ridge in MultiOutputRegressor")
        model = MultiOutputRegressor(base_regressor, n_jobs=-1)
    else:
        model = base_regressor

    print(
        f"Training RIDGE model with alpha={alpha} "
        f"on {X_train.shape[0]} samples, {X_train.shape[1]} features"
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"Ridge model training completed in {dt:.2f} s")

    return model


def train_dtree_model(X_train, y_train):
    """
    Train a Decision Tree Regressor model for correlator prediction.
    """
    validate_model_training_inputs(X_train, y_train)

    max_depth = getattr(config, "DTREE_MAX_DEPTH", 5)
    min_samples_leaf = getattr(config, "DTREE_MIN_SAMPLES_LEAF", 5)
    random_state = getattr(config, "RANDOM_SEED", 123)

    base_regressor = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    if getattr(config, "USE_MULTI_OUTPUT", True):
        print("Wrapping DecisionTree in MultiOutputRegressor")
        model = MultiOutputRegressor(base_regressor, n_jobs=-1)
    else:
        model = base_regressor

    print(
        "Training DTREE model with "
        f"max_depth={max_depth}, min_samples_leaf={min_samples_leaf} "
        f"on {X_train.shape[0]} samples, {X_train.shape[1]} features"
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"Decision Tree model training completed in {dt:.2f} s")

    return model


# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------


class _SimpleCNN1D(nn.Module):
    """
    Very small 1D CNN for correlator prediction.
    Treats the input vector as a 1D sequence with 1 channel.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        # x: (batch, input_dim)
        x = x.view(x.size(0), 1, self.input_dim)   # (batch, 1, T)
        x = self.conv(x)                           # (batch, 32, T)
        x = x.view(x.size(0), -1)                  # (batch, 32*T)
        x = self.fc(x)                             # (batch, output_dim)
        return x


def train_cnn_model(X_train, y_train):
    """
    Train a small 1D CNN for correlator prediction.
    """
    validate_model_training_inputs(X_train, y_train)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    n_epochs = getattr(config, "CNN_EPOCHS", 50)
    lr = getattr(config, "CNN_LR", 1e-3)
    batch_size = getattr(config, "CNN_BATCH_SIZE", 64)

    print(
        f"Training CNN model: input_dim={input_dim}, output_dim={output_dim}, "
        f"epochs={n_epochs}, lr={lr}, batch_size={batch_size}"
    )

    model = _SimpleCNN1D(input_dim, output_dim)
    reg = _TorchRegressorBase(model, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    reg.fit(X_train, y_train)
    return reg


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------


class _SimpleTransformerRegressor(nn.Module):
    """
    Small Transformer encoder for correlator prediction.

    We embed each scalar time slice into d_model, run a few encoder layers,
    pool over time, then map to output_dim.
    """

    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        # x: (batch, input_dim)
        bsz, T = x.shape
        x = x.view(bsz, T, 1)           # (batch, T, 1)
        x = self.input_proj(x)          # (batch, T, d_model)
        x = x.transpose(0, 1)           # (T, batch, d_model)
        enc = self.encoder(x)           # (T, batch, d_model)
        pooled = enc.mean(dim=0)        # (batch, d_model)
        out = self.fc(pooled)           # (batch, output_dim)
        return out


def train_transformer_model(X_train, y_train):
    """
    Train a small Transformer encoder for correlator prediction.
    """
    validate_model_training_inputs(X_train, y_train)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    d_model = getattr(config, "TRANSFORMER_D_MODEL", 64)
    nhead = getattr(config, "TRANSFORMER_NHEAD", 4)
    num_layers = getattr(config, "TRANSFORMER_NLAYERS", 2)
    n_epochs = getattr(config, "TRANSFORMER_EPOCHS", 50)
    lr = getattr(config, "TRANSFORMER_LR", 1e-3)
    batch_size = getattr(config, "TRANSFORMER_BATCH_SIZE", 64)

    print(
        "Training Transformer model: "
        f"input_dim={input_dim}, output_dim={output_dim}, d_model={d_model}, "
        f"nhead={nhead}, layers={num_layers}, epochs={n_epochs}, lr={lr}, "
        f"batch_size={batch_size}"
    )

    model = _SimpleTransformerRegressor(
        input_dim,
        output_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )
    reg = _TorchRegressorBase(model, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    reg.fit(X_train, y_train)
    return reg


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------


_MODEL_TRAINERS = {
    "GBR":         train_gbr_model,
    "MLP":         train_mlp_model,
    "RIDGE":       train_ridge_model,
    "DTREE":       train_dtree_model,
    "CNN":         train_cnn_model,
    "TRANSFORMER": train_transformer_model,
}


def train_model_by_name(model_name, X_train, y_train):
    """
    Train a model given its short name (e.g. 'GBR', 'MLP', 'RIDGE', 'DTREE',
    'CNN', 'TRANSFORMER').
    """
    key = model_name.upper()
    if key not in _MODEL_TRAINERS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options are: {sorted(_MODEL_TRAINERS.keys())}"
        )

    print(f"\n   -> Training {key} model via training.train_model_by_name")
    trainer = _MODEL_TRAINERS[key]
    return trainer(X_train, y_train)
