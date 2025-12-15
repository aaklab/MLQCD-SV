#!/usr/bin/env python3
"""
Data preparation and validation utilities for the Lattice QCD ML pipeline.
"""

import numpy as np
import config
from config import TRAIN_SOURCES, BC_SOURCE, UD_SOURCE
from sklearn.preprocessing import StandardScaler  # used by prepare_normalized_datasets


def symmetrise_correlators(C: np.ndarray) -> np.ndarray:
    """
    Enforce C(t) = C(T - t) by averaging each correlator with its time-reversed copy.

    C: array of shape (N_samples, N_t) or (..., N_t)
    Returns an array of the same shape.
    """
    return 0.5 * (C + C[..., ::-1])


def apply_time_window(C: np.ndarray, t_min: int, t_max: int) -> np.ndarray:
    """
    Restrict correlators to the time window [t_min, t_max).

    C: array of shape (N_samples, N_t) or (..., N_t)
    Returns an array of shape (N_samples, t_max - t_min).
    """
    return C[..., t_min:t_max]


def _moving_average_1d_last_axis(C: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply a simple moving average along the last axis of C.

    C: array of shape (..., N_t)
    kernel_size: odd integer >= 1
    """
    if kernel_size <= 1:
        return C

    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    # Pad along the last axis to keep output length N_t
    pad_width = kernel_size // 2
    pad_spec = [(0, 0)] * C.ndim
    pad_spec[-1] = (pad_width, pad_width)

    C_padded = np.pad(C, pad_spec, mode="edge")
    # Convolution along last axis
    kernel = np.ones(kernel_size, dtype=C.dtype) / kernel_size
    # Use np.apply_along_axis for clarity; performance is fine for our sizes
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"),
                               axis=-1, arr=C_padded)


def prepare_all_ml_datasets(input_data: np.ndarray,
                            target_data: np.ndarray,
                            n_configs: int):
    """
    High-level data preparation pipeline for the ML models.

    Steps (all controlled by config.py):
      1) Generic correlator preprocessing (symmetry, time window, etc.)
      2) Time-source partitioning (TRAIN / BC / UD)
      3) Build ML datasets (X_train, y_train, X_bc, y_bc, X_ud, y_ud)
      4) Optional input standardization with StandardScaler

    Returns:
      (input_reshaped, target_reshaped,
       X_train, y_train, X_bc, y_bc, X_ud, y_ud,
       train_indices, bc_indices, ud_indices,
       scaler)  # scaler can be None if standardization is off
    """

    # --- NEW SHAPE HANDLING BLOCK ---
    # input_data can be:
    #  - (n_cfg, n_sources, n_times), or
    #  - (n_cfg * n_sources, n_times)
    if input_data.ndim == 3:
        # Old behaviour
        n_cfg, n_sources, n_times = input_data.shape

    elif input_data.ndim == 2:
        # New behaviour: 2D array, need to recover n_cfg and n_sources
        n_rows, n_times = input_data.shape

        # train_indices / bc_indices / ud_indices are (cfg, src) pairs
        all_pairs = train_indices + bc_indices + ud_indices
        max_cfg = max(cfg for cfg, _ in all_pairs)
        max_src = max(src for _, src in all_pairs)

        n_cfg = max_cfg + 1
        n_sources = max_src + 1

        if n_cfg * n_sources != n_rows:
            raise ValueError(
                f"Shape mismatch in prepare_ml_datasets: "
                f"inferred n_cfg * n_sources = {n_cfg * n_sources}, "
                f"but got {n_rows} rows"
            )

        # Reshape back to (n_cfg, n_sources, n_times)
        input_data = input_data.reshape(n_cfg, n_sources, n_times)
        target_data = target_data.reshape(n_cfg, n_sources, n_times)

    else:
        raise ValueError(
        f"input_data must be 2D or 3D, got {input_data.ndim}D with shape {input_data.shape}"
        )

    # 1) Generic preprocessing
    input_reshaped, target_reshaped = preprocess_correlators(
        input_data,
        target_data,
    )

    # 2) Time-source partitions (uses TRAIN_SOURCES / BC_SOURCE / UD_SOURCE)
    train_indices, bc_indices, ud_indices = create_time_source_partitions(n_configs)

    # 3) Build ML datasets from the reshaped data
    X_train, y_train, X_bc, y_bc, X_ud, y_ud = prepare_ml_datasets(
        input_reshaped,
        target_reshaped,
        train_indices,
        bc_indices,
        ud_indices,
    )

    # 4) Optional input standardization
    scaler = None
    if getattr(config, "USE_INPUT_STANDARDIZATION", False):
        scaler, X_train, X_bc, X_ud = prepare_normalized_datasets(
            X_train, X_bc, X_ud
        )

    return (
        input_reshaped,
        target_reshaped,
        X_train,
        y_train,
        X_bc,
        y_bc,
        X_ud,
        y_ud,
        train_indices,
        bc_indices,
        ud_indices,
        scaler,
    )


def preprocess_correlators(input_data, target_data):
    """
    Unified preprocessing pipeline controlled by config.py.

    Steps (all optional, depending on config):
        1. Symmetry C(t) = C(T - t)                 [PRE_SYMMETRY]
        2. Time window [0, PRE_TIME_WINDOW_MAX)     [PRE_TIME_WINDOW]
        3. log(|C| + eps) transform                 [PRE_LOG_ABS, PRE_LOG_EPS]
        4. Normalisation ("none" / "l2" / "zscore") [PRE_NORMALISATION, PRE_NORMALISE_INPUT_ONLY]
        5. Control-variate ensemble-mean baseline   [CONTROL_VARIATE_ENABLED or CONTROL_VARIATE.ENABLED]

    Both input_data and target_data are 2D: (N_cfg * N_src, N_t_raw).

    Returns
    -------
    X, Y : np.ndarray
        Preprocessed input and target correlators,
        shape (N_cfg * N_src, N_t_windowed).
    """

    # Make safe float copies
    X = np.asarray(input_data, dtype=float).copy()
    Y = np.asarray(target_data, dtype=float).copy()

    # -----------------------------------------------------
    # 1. Symmetry C(t) = C(T - t)
    # -----------------------------------------------------
    if getattr(config, "PRE_SYMMETRY", True):
        print("[preprocess] Applying correlator symmetry C(t)=C(T-t)")
        X = symmetrise_correlators(X)
        Y = symmetrise_correlators(Y)

    # -----------------------------------------------------
    # 2. Time window [0, PRE_TIME_WINDOW_MAX)
    # -----------------------------------------------------
    if getattr(config, "PRE_TIME_WINDOW", True):
        Tmax = getattr(config, "PRE_TIME_WINDOW_MAX", X.shape[1])
        print(f"[preprocess] Applying time window [0, {Tmax})")
        X = X[:, :Tmax]
        Y = Y[:, :Tmax]

    # -----------------------------------------------------
    # 3. Optional log(|C| + eps) transform
    # -----------------------------------------------------
    if getattr(config, "PRE_LOG_ABS", False):
        eps = getattr(config, "PRE_LOG_EPS", 1e-12)
        print(f"[preprocess] Applying log(|C| + {eps}) transform")
        X = np.log(np.abs(X) + eps)
        Y = np.log(np.abs(Y) + eps)

    # -----------------------------------------------------
    # 4. Normalisation
    # -----------------------------------------------------
    mode = getattr(config, "PRE_NORMALISATION", "none").lower()
    norm_input_only = getattr(config, "PRE_NORMALISE_INPUT_ONLY", False)

    def _norm_l2(A):
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return A / norms

    def _norm_zscore(A):
        mean = A.mean(axis=0, keepdims=True)
        std = A.std(axis=0, ddof=1, keepdims=True)
        std[std == 0] = 1e-12
        return (A - mean) / std

    if mode == "l2":
        print("[preprocess] Applying row-wise L2 normalisation")
        X = _norm_l2(X)
        if not norm_input_only:
            Y = _norm_l2(Y)

    elif mode == "zscore":
        print("[preprocess] Applying per-time-slice z-score normalisation")
        X = _norm_zscore(X)
        if not norm_input_only:
            Y = _norm_zscore(Y)

    elif mode != "none":
        # Bad value – warn and skip
        print(f"[preprocess] WARNING: PRE_NORMALISATION='{mode}' not recognised; "
              "no normalisation applied")

    # -----------------------------------------------------
    # 5. Control-variate ensemble-mean baseline
    # -----------------------------------------------------
    # Support either flat flag CONTROL_VARIATE_ENABLED or a CONTROL_VARIATE namespace.
    cv_enabled = False
    if hasattr(config, "CONTROL_VARIATE_ENABLED"):
        cv_enabled = bool(getattr(config, "CONTROL_VARIATE_ENABLED"))
    elif hasattr(config, "CONTROL_VARIATE"):
        cv_enabled = bool(getattr(config.CONTROL_VARIATE, "ENABLED", False))

    if cv_enabled:
        print("[preprocess] Applying control-variate ensemble-mean baseline "
              "(subtracting mean correlator over samples)")
        X, _ = apply_control_variate_baseline(X, axis=0)
        Y, _ = apply_control_variate_baseline(Y, axis=0)

    return X, Y


def validate_array_dimensions(array, expected_shape, array_name):
    """
    Validate array dimensions and data types throughout the pipeline.
    
    Args:
        array (numpy.ndarray): Array to validate
        expected_shape (tuple): Expected shape of the array (use None for flexible dimensions)
        array_name (str): Name of the array for error messages
        
    Raises:
        TypeError: If array is not a numpy array
        ValueError: If array dimensions or data types are invalid
        
    Requirements: 8.5
    """
    # Check if input is a numpy array
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{array_name} must be a numpy array, got {type(array)}")
    
    # Check if array is empty
    if array.size == 0:
        raise ValueError(f"{array_name} cannot be empty")
    
    # Check data type - must be numeric
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{array_name} must contain numeric data, got dtype {array.dtype}")
    
    # Check for NaN or infinite values
    if np.isnan(array).any():
        raise ValueError(f"{array_name} contains NaN values")
    
    if np.isinf(array).any():
        raise ValueError(f"{array_name} contains infinite values")
    
    # Check dimensions
    if expected_shape is not None:
        if len(array.shape) != len(expected_shape):
            raise ValueError(f"{array_name} must have {len(expected_shape)} dimensions, got {len(array.shape)}")
        
        for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValueError(f"{array_name} dimension {i} must be {expected}, got {actual}")
    
    # Check for reasonable value ranges (correlator data should not be extremely large)
    max_abs_value = np.max(np.abs(array))
    if max_abs_value > 1e10:
        raise ValueError(f"{array_name} contains extremely large values (max: {max_abs_value:.2e})")
    
    # Check for extremely small values that might indicate precision issues
    # Note: Lattice QCD correlator data can legitimately have very small values at large time separations
    min_abs_nonzero = np.min(np.abs(array[array != 0])) if np.any(array != 0) else 0
    if min_abs_nonzero > 0 and min_abs_nonzero < 1e-30:
        print(f"Warning: {array_name} contains very small non-zero values (min: {min_abs_nonzero:.2e}) - this may be normal for correlator data at large time separations")


def validate_correlator_data_structure(input_data, target_data, n_configs, n_sources, n_times):
    """
    Validate the structure of correlator data arrays for consistency.
    
    Args:
        input_data (numpy.ndarray): Input correlator data
        target_data (numpy.ndarray): Target correlator data
        n_configs (int): Expected number of configurations
        n_sources (int): Expected number of time sources per configuration
        n_times (int): Expected number of time slices
        
    Raises:
        ValueError: If data structure is inconsistent or invalid
        
    Requirements: 8.5
    """
    # Validate individual arrays
    validate_array_dimensions(input_data, (n_configs * n_sources, n_times), "input_data")
    validate_array_dimensions(target_data, (n_configs * n_sources, n_times), "target_data")
    
    # Check that input and target have the same shape
    if input_data.shape != target_data.shape:
        raise ValueError(f"Input and target data must have same shape. Got input: {input_data.shape}, target: {target_data.shape}")
    
    # Validate configuration parameters
    if n_configs <= 0:
        raise ValueError(f"Number of configurations must be positive, got {n_configs}")
    
    if n_sources != config.N_TIME_SOURCES:
        raise ValueError(f"Number of time sources must be {config.N_TIME_SOURCES}, got {n_sources}")
    
    if n_times <= 0:
        raise ValueError(f"Number of time slices must be positive, got {n_times}")
    
    # Check that the total number of rows matches expected structure
    expected_rows = n_configs * n_sources
    actual_rows = input_data.shape[0]
    if actual_rows != expected_rows:
        raise ValueError(f"Data rows ({actual_rows}) must equal n_configs × n_sources ({expected_rows})")
    
    # Check time dimension consistency
    if input_data.shape[1] != n_times or target_data.shape[1] != n_times:
        raise ValueError(f"Data must have {n_times} time slices, got input: {input_data.shape[1]}, target: {target_data.shape[1]}")


def validate_partition_indices(train_indices, bc_indices, ud_indices, n_configs, n_sources):
    """
    Validate time source partition indices for completeness and correctness.
    
    Args:
        train_indices (list): List of (cfg, src) pairs for training
        bc_indices (list): List of (cfg, src) pairs for bias correction
        ud_indices (list): List of (cfg, src) pairs for unbiased validation
        n_configs (int): Number of configurations
        n_sources (int): Number of time sources per configuration
        
    Raises:
        ValueError: If partition indices are invalid or incomplete
        
    Requirements: 8.5
    """
    # Check that all indices are lists
    for name, indices in [("train_indices", train_indices), ("bc_indices", bc_indices), ("ud_indices", ud_indices)]:
        if not isinstance(indices, list):
            raise ValueError(f"{name} must be a list, got {type(indices)}")
    
    # Check expected sizes
    expected_train_size = n_configs * len(config.TRAIN_SOURCES)  # 2 sources per config
    expected_bc_size = n_configs * len(config.BC_SOURCE)         # 1 source per config
    expected_ud_size = n_configs * len(config.UD_SOURCE)         # 1 source per config
    
    if len(train_indices) != expected_train_size:
        raise ValueError(f"train_indices must have {expected_train_size} elements, got {len(train_indices)}")
    
    if len(bc_indices) != expected_bc_size:
        raise ValueError(f"bc_indices must have {expected_bc_size} elements, got {len(bc_indices)}")
    
    if len(ud_indices) != expected_ud_size:
        raise ValueError(f"ud_indices must have {expected_ud_size} elements, got {len(ud_indices)}")
    
    # Validate individual indices
    all_indices = train_indices + bc_indices + ud_indices
    for i, (cfg, src) in enumerate(all_indices):
        if not isinstance(cfg, int) or not isinstance(src, int):
            raise ValueError(f"Index {i} must be a tuple of integers, got ({type(cfg)}, {type(src)})")
        
        if cfg < 0 or cfg >= n_configs:
            raise ValueError(f"Configuration index {cfg} out of range [0, {n_configs})")
        
        if src < 0 or src >= n_sources:
            raise ValueError(f"Source index {src} out of range [0, {n_sources})")
    
    # Check for completeness - every (cfg, src) pair should appear exactly once
    expected_pairs = set((cfg, src) for cfg in range(n_configs) for src in range(n_sources))
    actual_pairs = set(all_indices)
    
    if actual_pairs != expected_pairs:
        missing = expected_pairs - actual_pairs
        extra = actual_pairs - expected_pairs
        
        error_msg = "Partition indices are not complete and disjoint."
        if missing:
            error_msg += f" Missing: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        if extra:
            error_msg += f" Extra: {list(extra)[:5]}{'...' if len(extra) > 5 else ''}"
        
        raise ValueError(error_msg)
    
    # Validate partition assignments
    train_pairs = set(train_indices)
    bc_pairs = set(bc_indices)
    ud_pairs = set(ud_indices)
    
    # Check that partitions are disjoint
    if train_pairs & bc_pairs:
        raise ValueError("TRAIN and BC partitions must be disjoint")
    if train_pairs & ud_pairs:
        raise ValueError("TRAIN and UD partitions must be disjoint")
    if bc_pairs & ud_pairs:
        raise ValueError("BC and UD partitions must be disjoint")
    
    # Check that partitions follow the expected pattern
    for cfg in range(n_configs):
        expected_train = {(cfg, src) for src in TRAIN_SOURCES}
        expected_bc = {(cfg, src) for src in BC_SOURCE}
        expected_ud = {(cfg, src) for src in UD_SOURCE}
        
        actual_train = {(c, s) for (c, s) in train_pairs if c == cfg}
        actual_bc = {(c, s) for (c, s) in bc_pairs if c == cfg}
        actual_ud = {(c, s) for (c, s) in ud_pairs if c == cfg}
        
        if actual_train != expected_train:
            raise ValueError(f"Config {cfg} TRAIN partition incorrect. Expected: {expected_train}, got: {actual_train}")
        if actual_bc != expected_bc:
            raise ValueError(f"Config {cfg} BC partition incorrect. Expected: {expected_bc}, got: {actual_bc}")
        if actual_ud != expected_ud:
            raise ValueError(f"Config {cfg} UD partition incorrect. Expected: {expected_ud}, got: {actual_ud}")
        

def validate_statistical_computation_inputs(data_arrays, method_names):
    """
    Validate inputs for statistical computations and handle edge cases.
    
    Args:
        data_arrays (list): List of numpy arrays to validate
        method_names (list): List of method names corresponding to the arrays
        
    Raises:
        ValueError: If inputs are invalid for statistical computation
        
    Requirements: 8.5
    """
    if len(data_arrays) != len(method_names):
        raise ValueError(f"Number of data arrays ({len(data_arrays)}) must match number of method names ({len(method_names)})")
    
    if len(data_arrays) == 0:
        raise ValueError("At least one data array must be provided")
    
    # Validate each array
    reference_shape = None
    for i, (array, name) in enumerate(zip(data_arrays, method_names)):
        validate_array_dimensions(array, None, f"{name}_data")
        
        # Check that array is 2D (configurations × time)
        if len(array.shape) != 2:
            raise ValueError(f"{name}_data must be 2D (configurations × time), got shape {array.shape}")
        
        # Check that all arrays have the same shape
        if reference_shape is None:
            reference_shape = array.shape
        elif array.shape != reference_shape:
            raise ValueError(f"All data arrays must have the same shape. {method_names[0]}: {reference_shape}, {name}: {array.shape}")
        
        # Check for sufficient configurations for meaningful statistics
        n_configs = array.shape[0]
        if n_configs < 2:
            raise ValueError(f"Need at least 2 configurations for statistics, got {n_configs}")
        
        # Check for edge cases in statistical computation
        # Warn about configurations where all values are identical (zero variance)
        for t in range(array.shape[1]):
            time_slice = array[:, t]
            if np.all(time_slice == time_slice[0]):
                print(f"Warning: {name} has identical values across all configurations at time slice {t}")
        
        # Check for extremely small values that might cause numerical issues
        abs_values = np.abs(array)
        min_nonzero = np.min(abs_values[abs_values > 0]) if np.any(abs_values > 0) else 0
        if min_nonzero > 0 and min_nonzero < 1e-12:
            print(f"Warning: {name} contains very small values (min non-zero: {min_nonzero:.2e}) that may cause numerical precision issues")
        
        # Check for values that are exactly zero (which will cause issues in NtS calculation)
        zero_means = np.sum(np.abs(np.mean(array, axis=0)) < 1e-15)
        if zero_means > 0:
            print(f"Warning: {name} has {zero_means} time slices with near-zero ensemble means, NtS ratios will be infinite")

def reshape_correlator_data(data, n_configs, n_sources, n_times):
    """
    Reshape correlator data from (N_rows, N_t) to (N_cfg, 4, N_t) format.
    
    Args:
        data (numpy.ndarray): Input data of shape (N_rows, N_t)
        n_configs (int): Number of configurations (N_cfg)
        n_sources (int): Number of time sources per configuration (should be 4)
        n_times (int): Number of Euclidean time slices (N_t)
        
    Returns:
        numpy.ndarray: Reshaped data of shape (N_cfg, 4, N_t)
        
    Raises:
        ValueError: If N_rows != N_cfg * 4 or if dimensions are inconsistent
        
    Requirements: 1.3, 1.4
    """
    # Validate input array dimensions and data types
    validate_array_dimensions(data, (n_configs * n_sources, n_times), "input_data")
    
    # Validate configuration parameters
    if n_configs <= 0:
        raise ValueError(f"Number of configurations must be positive, got {n_configs}")
    
    if n_sources != config.N_TIME_SOURCES:
        raise ValueError(f"Expected {config.N_TIME_SOURCES} time sources, got {n_sources}")
    
    if n_times <= 0:
        raise ValueError(f"Number of time slices must be positive, got {n_times}")
    
    # Reshape from (N_rows, N_t) to (N_cfg, 4, N_t)
    reshaped_data = data.reshape(n_configs, n_sources, n_times)
    
    # Validate output shape
    validate_array_dimensions(reshaped_data, (n_configs, n_sources, n_times), "reshaped_data")
    
    return reshaped_data


def create_time_source_partitions(n_cfg, n_sources=4):
    """
    Create time source partition indices for training, bias correction, and validation.
    
    Args:
        n_cfg (int): Number of configurations
        n_sources (int): Number of time sources per configuration (default: 4)
        
    Returns:
        tuple: (train_indices, bc_indices, ud_indices)
            - train_indices: List of (cfg, src) pairs for TRAIN sources {0, 1}
            - bc_indices: List of (cfg, src) pairs for BC source {2}
            - ud_indices: List of (cfg, src) pairs for UD source {3}
            
    Requirements: 2.1, 2.2, 2.3, 2.4
    """
    # Validate input parameters
    if n_cfg <= 0:
        raise ValueError(f"Number of configurations must be positive, got {n_cfg}")
    
    if n_sources != config.N_TIME_SOURCES:
        raise ValueError(f"Expected {config.N_TIME_SOURCES} time sources, got {n_sources}")
    
    # Initialize partition lists
    train_indices = []
    bc_indices = []
    ud_indices = []
    
    # Create partition indices according to the specification:
    # TRAIN = {0, 1}, BC = {2}, UD = {3}
    for cfg in range(n_cfg):
        # TRAIN sources: {0, 1}
        for src in TRAIN_SOURCES:
            train_indices.append((cfg, src))
        
        # BC source: {2}
        for src in BC_SOURCE:
            bc_indices.append((cfg, src))
            
        # UD source: {3}  
        for src in UD_SOURCE:
            ud_indices.append((cfg, src))
    
    # Validate the created partitions
    validate_partition_indices(train_indices, bc_indices, ud_indices, n_cfg, n_sources)
    
    return train_indices, bc_indices, ud_indices


def prepare_ml_datasets(input_data, target_data, train_indices, bc_indices, ud_indices):
    """
    Build training datasets using partition indices for machine learning models.
    
    Args:
        input_data (numpy.ndarray): Input correlator data of shape (N_cfg, 4, N_t)
        target_data (numpy.ndarray): Target correlator data of shape (N_cfg, 4, N_t)
        train_indices (list): List of (cfg, src) pairs for training data
        bc_indices (list): List of (cfg, src) pairs for bias correction
        ud_indices (list): List of (cfg, src) pairs for unbiased validation
        
    Returns:
        tuple: (X_train, y_train, X_bc, y_bc, X_ud, y_ud)
            - X_train: Training input data of shape (N_train_samples, N_t)
            - y_train: Training target data of shape (N_train_samples, N_t)
            - X_bc: Bias correction input data of shape (N_cfg, N_t)
            - y_bc: Bias correction target data of shape (N_cfg, N_t)
            - X_ud: Unbiased validation input data of shape (N_cfg, N_t)
            - y_ud: Unbiased validation target data of shape (N_cfg, N_t)
            
    Requirements: 2.5
    """
    # --- NEW SHAPE HANDLING BLOCK ---
    # input_data can be:
    #  - (n_cfg, n_sources, n_times), or
    #  - (n_cfg * n_sources, n_times)
    if input_data.ndim == 3:
        # Old behaviour
        n_cfg, n_sources, n_times = input_data.shape

    elif input_data.ndim == 2:
        # New behaviour: 2D array, need to recover n_cfg and n_sources
        n_rows, n_times = input_data.shape

        # train_indices / bc_indices / ud_indices are (cfg, src) pairs
        all_pairs = train_indices + bc_indices + ud_indices
        max_cfg = max(cfg for cfg, _ in all_pairs)
        max_src = max(src for _, src in all_pairs)

        n_cfg = max_cfg + 1
        n_sources = max_src + 1

        if n_cfg * n_sources != n_rows:
            raise ValueError(
                f"Shape mismatch in prepare_ml_datasets: "
                f"inferred n_cfg * n_sources = {n_cfg * n_sources}, "
                f"but got {n_rows} rows"
            )

        # Reshape back to (n_cfg, n_sources, n_times)
        input_data = input_data.reshape(n_cfg, n_sources, n_times)
        target_data = target_data.reshape(n_cfg, n_sources, n_times)

    else:
        raise ValueError(
        f"input_data must be 2D or 3D, got {input_data.ndim}D with shape {input_data.shape}"
        )
    
    # Validate input data structure
    validate_correlator_data_structure(
        input_data.reshape(-1, n_times), 
        target_data.reshape(-1, n_times), 
        n_cfg, n_sources, n_times
    )
    
    # Validate partition indices
    validate_partition_indices(train_indices, bc_indices, ud_indices, n_cfg, n_sources)
    
    # Build training dataset from TRAIN indices
    X_train_list = []
    y_train_list = []
    
    for cfg, src in train_indices:
        X_train_list.append(input_data[cfg, src, :])
        y_train_list.append(target_data[cfg, src, :])
    
    X_train = np.array(X_train_list)  # Shape: (N_train_samples, N_t)
    y_train = np.array(y_train_list)  # Shape: (N_train_samples, N_t)
    
    # Build bias correction dataset from BC indices
    X_bc_list = []
    y_bc_list = []
    
    for cfg, src in bc_indices:
        X_bc_list.append(input_data[cfg, src, :])
        y_bc_list.append(target_data[cfg, src, :])
    
    X_bc = np.array(X_bc_list)  # Shape: (N_cfg, N_t)
    y_bc = np.array(y_bc_list)  # Shape: (N_cfg, N_t)
    
    # Build unbiased validation dataset from UD indices
    X_ud_list = []
    y_ud_list = []
    
    for cfg, src in ud_indices:
        X_ud_list.append(input_data[cfg, src, :])
        y_ud_list.append(target_data[cfg, src, :])
    
    X_ud = np.array(X_ud_list)  # Shape: (N_cfg, N_t)
    y_ud = np.array(y_ud_list)  # Shape: (N_cfg, N_t)
    
    # Validate output arrays
    expected_train_samples = len(train_indices)
    expected_bc_samples = len(bc_indices)
    expected_ud_samples = len(ud_indices)
    
    validate_array_dimensions(X_train, (expected_train_samples, n_times), "X_train")
    validate_array_dimensions(y_train, (expected_train_samples, n_times), "y_train")
    validate_array_dimensions(X_bc, (expected_bc_samples, n_times), "X_bc")
    validate_array_dimensions(y_bc, (expected_bc_samples, n_times), "y_bc")
    validate_array_dimensions(X_ud, (expected_ud_samples, n_times), "X_ud")
    validate_array_dimensions(y_ud, (expected_ud_samples, n_times), "y_ud")
    
    return X_train, y_train, X_bc, y_bc, X_ud, y_ud


def prepare_normalized_datasets(X_train, X_bc, X_ud):
    """
    Apply StandardScaler normalization to input correlator datasets to prevent model collapse.
    
    This function implements mandatory input normalization as specified in Requirements 9.1 and 9.2.
    The StandardScaler is fitted on training data and applied consistently to all datasets
    to ensure proper scaling and prevent MLP model collapse.
    
    Args:
        X_train (numpy.ndarray): Training input data of shape (N_train_samples, N_features)
        X_bc (numpy.ndarray): Bias correction input data of shape (N_bc_samples, N_features)
        X_ud (numpy.ndarray): Unbiased validation input data of shape (N_ud_samples, N_features)
        
    Returns:
        tuple: (scaler, X_train_scaled, X_bc_scaled, X_ud_scaled)
            - scaler: Fitted StandardScaler object for potential inverse transforms
            - X_train_scaled: Normalized training data
            - X_bc_scaled: Normalized bias correction data  
            - X_ud_scaled: Normalized unbiased validation data
            
    Requirements: 9.1, 9.2
    """
    # Validate input arrays
    validate_array_dimensions(X_train, None, "X_train")
    validate_array_dimensions(X_bc, None, "X_bc")
    validate_array_dimensions(X_ud, None, "X_ud")
    
    # Check that all arrays have the same number of features
    n_features_train = X_train.shape[1]
    n_features_bc = X_bc.shape[1]
    n_features_ud = X_ud.shape[1]
    
    if not (n_features_train == n_features_bc == n_features_ud):
        raise ValueError(f"All datasets must have same number of features. "
                        f"Got train: {n_features_train}, bc: {n_features_bc}, ud: {n_features_ud}")
    
    # Initialize and fit StandardScaler on training data only
    scaler = StandardScaler()
    
    print(f"Fitting StandardScaler on training data with shape {X_train.shape}")
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply the same scaler to BC and UD datasets
    print(f"Applying scaler to BC data with shape {X_bc.shape}")
    X_bc_scaled = scaler.transform(X_bc)
    
    print(f"Applying scaler to UD data with shape {X_ud.shape}")
    X_ud_scaled = scaler.transform(X_ud)
    
    # Validate that scaling was applied correctly
    validate_array_dimensions(X_train_scaled, X_train.shape, "X_train_scaled")
    validate_array_dimensions(X_bc_scaled, X_bc.shape, "X_bc_scaled")
    validate_array_dimensions(X_ud_scaled, X_ud.shape, "X_ud_scaled")
    
    # Verify that training data is properly normalized (mean ≈ 0, std ≈ 1)
    train_means = np.mean(X_train_scaled, axis=0)
    train_stds = np.std(X_train_scaled, axis=0)
    
    # Check that means are close to 0 and stds are close to 1
    if not np.allclose(train_means, 0, atol=1e-10):
        print(f"Warning: Training data means after scaling are not close to 0: {np.max(np.abs(train_means)):.2e}")
    
    if not np.allclose(train_stds, 1, atol=1e-10):
        print(f"Warning: Training data stds after scaling are not close to 1: {np.max(np.abs(train_stds - 1)):.2e}")
    
    print("Input normalization completed successfully")
    print(f"Training data: mean={np.mean(train_means):.2e}, std={np.mean(train_stds):.3f}")
    
    return scaler, X_train_scaled, X_bc_scaled, X_ud_scaled


def apply_control_variate_baseline(y: np.ndarray, axis: int = 0):
    """
    Apply a control-variate style preprocessing to correlators.

    Parameters
    ----------
    y : np.ndarray
        Target array of shape (n_samples, T, ...) containing correlators.
        We assume the 'sample' dimension is `axis` (usually 0).
    axis : int
        Axis along which to compute the baseline (typically sample axis).

    Returns
    -------
    y_residual : np.ndarray
        Residual targets: y - baseline
    baseline : np.ndarray
        Baseline correlator(s) with same shape as y, broadcast along `axis`.
        You must store this to be able to reconstruct predictions later.
    """
    # Move sample axis to front for simplicity
    y_moved = np.moveaxis(y, axis, 0)  # shape: (N, ...)

    # Ensemble-mean baseline across configurations
    baseline_core = y_moved.mean(axis=0, keepdims=True)  # shape: (1, ...)

    # Broadcast subtraction
    y_residual_moved = y_moved - baseline_core

    # Move axes back to original order
    y_residual = np.moveaxis(y_residual_moved, 0, axis)

    # Broadcast `baseline_core` back to full y shape
    baseline_full = np.broadcast_to(baseline_core, y_moved.shape)
    baseline_full = np.moveaxis(baseline_full, 0, axis)

    return y_residual, baseline_full