#!/usr/bin/env python3
"""
Lattice QCD Analysis Pipeline

This script implements a machine learning-based approach to improve signal-to-noise ratios 
in 2-point correlator measurements using bias-corrected estimators with Gradient Boosting 
Regressor (GBR) and Multi-Layer Perceptron (MLP) models.

Requirements: 7.2, 7.4, 8.4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random
import os

# Experiment Configuration Dictionary
EXPERIMENTS = {
    "1": {
        "label": "K_ll_to_K_qsq0",
        "description": "Kaon: ll → qsq0",
        "input_csv": "data/raw/2pt_K_fine_ll.csv",
        "target_csv": "data/raw/2pt_K_fine_qsq0_ll.csv",
    },
    "2": {
        "label": "K_ll_to_K_qsqmaxby3",
        "description": "Kaon: ll → qsqmaxby3",
        "input_csv": "data/raw/2pt_K_fine_ll.csv",
        "target_csv": "data/raw/2pt_K_fine_qsqmaxby3_ll.csv",
    },
    "3": {
        "label": "K_ll_to_K_2qsqmaxby3",
        "description": "Kaon: ll → 2qsqmaxby3",
        "input_csv": "data/raw/2pt_K_fine_ll.csv",
        "target_csv": "data/raw/2pt_K_fine_2qsqmaxby3_ll.csv",
    },
    "4": {
        "label": "D_gold_to_D_nongold",
        "description": "D-meson: Gold → non-Gold",
        "input_csv": "data/raw/2pt_D_Gold_fine_ll.csv",
        "target_csv": "data/raw/2pt_D_nongold_fine_ll.csv",
    },
}

# Set random seeds for reproducible results (Requirement 8.4)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Global constants for time source partitioning (Requirements 2.1, 2.2, 2.3, 2.4)
TRAIN_SOURCES = [0, 1]  # Training data time sources
BC_SOURCE = [2]         # Bias correction time source
UD_SOURCE = [3]         # Unbiased validation time source
N_TIME_SOURCES = 4      # Total number of time sources per configuration

# Constants for improved bias-correction plots
TAU_MIN = 5             # Minimum tau for bias-correction plots
TAU_MAX = 60            # Maximum tau for bias-correction plots
TRUTH_MAGNITUDE_THRESHOLD = 1e-7  # Threshold for masking small truth values
BIAS_PLOT_Y_LIMITS = (-10, 10)   # Y-axis limits for bias-correction plots

# Scientific plotting configuration
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


# Experiment Selection Module

def choose_experiment():
    """
    Interactive function to choose which 2-point correlator experiment to run.
    
    Returns:
        dict: Selected experiment configuration containing label, description, 
              input_csv, and target_csv paths
    """
    print("Available 2-point correlator experiments:")
    for key, cfg in sorted(EXPERIMENTS.items()):
        print(f"  {key}) {cfg['description']} "
              f"[input: {cfg['input_csv'].split('/')[-1]}, "
              f"target: {cfg['target_csv'].split('/')[-1]}]")
    
    choice = None
    while choice not in EXPERIMENTS:
        choice = input("Select an experiment (1–4): ").strip()
        if choice not in EXPERIMENTS:
            print(f"Invalid choice '{choice}'. Please select 1, 2, 3, or 4.")
    
    cfg = EXPERIMENTS[choice]
    print(f"\nSelected experiment: {cfg['description']}")
    print(f"  Input CSV : {cfg['input_csv']}")
    print(f"  Target CSV: {cfg['target_csv']}")
    print(f"  Label     : {cfg['label']}")
    
    return cfg


# Data Validation Module

def validate_csv_file_format(file_path):
    """
    Validate CSV file format and expected data structure for correlator data.
    
    Args:
        file_path (str): Path to the CSV file to validate
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is invalid or data structure is unexpected
        
    Requirements: 8.5
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        raise ValueError(f"CSV file is not readable: {file_path}")
    
    # Check file extension
    if not file_path.lower().endswith('.csv'):
        raise ValueError(f"File must have .csv extension: {file_path}")
    
    try:
        # Try to read the CSV file
        data = pd.read_csv(file_path, header=None)
        
        # Validate that we have numeric data
        if data.empty:
            raise ValueError(f"CSV file is empty: {file_path}")
        
        # Check that all data is numeric
        if not data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]:
            raise ValueError(f"CSV file contains non-numeric data: {file_path}")
        
        # Check for NaN or infinite values
        if data.isnull().any().any():
            raise ValueError(f"CSV file contains NaN values: {file_path}")
        
        if np.isinf(data.values).any():
            raise ValueError(f"CSV file contains infinite values: {file_path}")
        
        # Validate minimum dimensions
        if data.shape[0] < N_TIME_SOURCES:
            raise ValueError(f"CSV file must have at least {N_TIME_SOURCES} rows (configurations × time sources): {file_path}")
        
        if data.shape[1] < 1:
            raise ValueError(f"CSV file must have at least 1 column (time slices): {file_path}")
        
        # Check that we have enough rows for at least one complete configuration
        if data.shape[0] < N_TIME_SOURCES:
            raise ValueError(f"CSV file must have at least {N_TIME_SOURCES} rows for one complete configuration: {file_path}")
        
        # Calculate number of complete configurations (may trim incomplete ones)
        n_complete_configs = data.shape[0] // N_TIME_SOURCES
        if data.shape[0] % N_TIME_SOURCES != 0:
            print(f"Warning: CSV file has {data.shape[0]} rows, will use {n_complete_configs * N_TIME_SOURCES} rows ({n_complete_configs} complete configurations)")
        
        print(f"CSV file validation passed: {file_path} ({data.shape[0]} rows, {data.shape[1]} columns, {n_complete_configs} complete configs)")
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty or corrupted: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV file parsing error: {file_path} - {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error reading CSV file: {file_path} - {str(e)}")


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
    
    if n_sources != N_TIME_SOURCES:
        raise ValueError(f"Number of time sources must be {N_TIME_SOURCES}, got {n_sources}")
    
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
    expected_train_size = n_configs * len(TRAIN_SOURCES)  # 2 sources per config
    expected_bc_size = n_configs * len(BC_SOURCE)         # 1 source per config
    expected_ud_size = n_configs * len(UD_SOURCE)         # 1 source per config
    
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


def validate_model_training_inputs(X_train, y_train):
    """
    Validate inputs for machine learning model training.
    
    Args:
        X_train (numpy.ndarray): Training input data
        y_train (numpy.ndarray): Training target data
        
    Raises:
        ValueError: If training data is invalid
        
    Requirements: 8.5
    """
    # Validate array types and basic properties
    validate_array_dimensions(X_train, None, "X_train")
    validate_array_dimensions(y_train, None, "y_train")
    
    # Check that both arrays are 2D
    if len(X_train.shape) != 2:
        raise ValueError(f"X_train must be 2D (samples × features), got shape {X_train.shape}")
    
    if len(y_train.shape) != 2:
        raise ValueError(f"y_train must be 2D (samples × outputs), got shape {y_train.shape}")
    
    # Check that number of samples match
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train and y_train must have same number of samples. Got {X_train.shape[0]} and {y_train.shape[0]}")
    
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
            raise ValueError(f"Feature {i} in X_train has no variation (all values are {feature[0]})")
    
    # Check for constant targets (no variation to learn)
    for i in range(y_train.shape[1]):
        target = y_train[:, i]
        if np.all(target == target[0]):
            print(f"Warning: Target {i} in y_train has no variation (all values are {target[0]})")
    
    print(f"Model training input validation passed: {n_samples} samples, {X_train.shape[1]} features, {y_train.shape[1]} outputs")


# Data Loading and Preprocessing Module

def load_correlator_data(input_path, target_path):
    """
    Load correlator data from CSV files and extract truth values from column names.
    
    Args:
        input_path (str): Path to input correlator CSV file
        target_path (str): Path to target correlator CSV file
        
    Returns:
        tuple: (input_data, target_data, truth_input, truth_target, time_values)
            - input_data: numpy array of shape (N_rows, N_t) with input correlator data
            - target_data: numpy array of shape (N_rows, N_t) with target correlator data  
            - truth_input: float value extracted from input filename
            - truth_target: float value extracted from target filename
            - time_values: numpy array of time slice indices [0, 1, 2, ..., N_t-1]
    
    Requirements: 1.1, 1.2
    """
    # Validate CSV file formats before loading
    validate_csv_file_format(input_path)
    validate_csv_file_format(target_path)
    
    # Load CSV data without headers (correlator data is numeric only)
    input_data_raw = pd.read_csv(input_path, header=None).values
    target_data_raw = pd.read_csv(target_path, header=None).values
    
    # Ensure both datasets have the same number of configurations
    input_rows = input_data_raw.shape[0]
    target_rows = target_data_raw.shape[0]
    
    # Calculate number of complete configurations for each dataset
    input_configs = input_rows // N_TIME_SOURCES
    target_configs = target_rows // N_TIME_SOURCES
    
    # Use the minimum number of configurations to ensure compatibility
    n_configs = min(input_configs, target_configs)
    n_rows = n_configs * N_TIME_SOURCES
    
    # Trim data to have matching number of configurations
    input_data = input_data_raw[:n_rows]
    target_data = target_data_raw[:n_rows]
    
    # Extract truth correlator values from filenames by converting to float
    # For files like "2pt_K_fine_ll.csv" and "2pt_K_fine_qsq0_ll.csv"
    # Extract the numeric part that represents the truth correlator value
    input_basename = os.path.basename(input_path)
    target_basename = os.path.basename(target_path)
    
    # For input file (e.g., "2pt_K_fine_ll.csv"), truth value is typically 0.0
    # For target file (e.g., "2pt_K_fine_qsq0_ll.csv"), extract qsq value
    if "qsq0" in target_basename:
        truth_target = 0.0
    elif "qsqmaxby3" in target_basename:
        truth_target = 1.0/3.0  # qsq_max/3
    elif "2qsqmaxby3" in target_basename:
        truth_target = 2.0/3.0  # 2*qsq_max/3
    else:
        # Default case - extract any numeric value from filename
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', target_basename)
        truth_target = float(numbers[0]) if numbers else 0.0
    
    # Input truth value (typically 0.0 for base correlator)
    if "qsq0" in input_basename:
        truth_input = 0.0
    elif "qsqmaxby3" in input_basename:
        truth_input = 1.0/3.0
    elif "2qsqmaxby3" in input_basename:
        truth_input = 2.0/3.0
    else:
        truth_input = 0.0  # Default for base correlator
    
    # Create time values array [0, 1, 2, ..., N_t-1]
    n_times = input_data.shape[1]
    time_values = np.arange(n_times)
    
    return input_data, target_data, truth_input, truth_target, time_values


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
    
    if n_sources != N_TIME_SOURCES:
        raise ValueError(f"Expected {N_TIME_SOURCES} time sources, got {n_sources}")
    
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
    
    if n_sources != N_TIME_SOURCES:
        raise ValueError(f"Expected {N_TIME_SOURCES} time sources, got {n_sources}")
    
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


# Machine Learning Module

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
    n_cfg, n_sources, n_times = input_data.shape
    
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


def train_gbr_model(X_train, y_train):
    """
    Train a Gradient Boosting Regressor model for correlator prediction.
    
    Args:
        X_train (numpy.ndarray): Training input data of shape (N_samples, N_features)
        y_train (numpy.ndarray): Training target data of shape (N_samples, N_outputs)
        
    Returns:
        sklearn.multioutput.MultiOutputRegressor: Trained GBR model wrapped in MultiOutputRegressor
        
    Requirements: 3.1, 3.3
    """
    # Validate model training inputs
    validate_model_training_inputs(X_train, y_train)
    
    # Configure GradientBoostingRegressor with reduced complexity for faster training
    # These parameters are chosen to handle the exponential decay nature of correlator data
    gbr_base = GradientBoostingRegressor(
        n_estimators=50,            # Reduced number of boosting stages for faster training
        learning_rate=0.1,          # Learning rate shrinks contribution of each tree
        max_depth=4,                # Reduced depth for faster training
        min_samples_split=5,        # Increased to reduce overfitting and speed up training
        min_samples_leaf=2,         # Increased to reduce overfitting
        subsample=0.8,              # Fraction of samples used for fitting individual base learners
        random_state=RANDOM_SEED,   # Random state for reproducibility
        loss='squared_error',       # Loss function for regression
        alpha=0.9                   # Alpha-quantile of the huber loss function
    )
    
    # Wrap in MultiOutputRegressor to handle multiple time slice outputs
    gbr_model = MultiOutputRegressor(gbr_base, n_jobs=1)
    
    # Train the model
    print(f"Training GBR model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    print(f"Target shape: {y_train.shape} (predicting {y_train.shape[1]} time slices)")
    
    gbr_model.fit(X_train, y_train)
    
    print("GBR model training completed successfully")
    
    return gbr_model


def estimate_mlp_parameters(hidden_layer_sizes, n_features, n_outputs):
    """
    Estimate the total number of parameters in an MLP model.
    
    Args:
        hidden_layer_sizes (tuple): Hidden layer sizes
        n_features (int): Number of input features
        n_outputs (int): Number of output neurons
        
    Returns:
        int: Estimated total number of parameters
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


def train_mlp_model(X_train, y_train):
    """
    Train a Multi-Layer Perceptron model for correlator prediction with improved configuration
    to prevent model collapse. Uses a single MLPRegressor with Pipeline for proper scaling.
    
    CRITICAL FIX: Removes MultiOutputRegressor which was causing model collapse by training
    separate models for each time slice. Instead uses a single MLP that outputs all time slices.
    
    Args:
        X_train (numpy.ndarray): Training input data of shape (N_samples, N_features)
        y_train (numpy.ndarray): Training target data of shape (N_samples, N_outputs)
        
    Returns:
        sklearn.pipeline.Pipeline: Trained MLP model with StandardScaler preprocessing
        
    Requirements: 3.2, 3.3, 9.3, 9.4, 9.5
    """
    # Validate model training inputs
    validate_model_training_inputs(X_train, y_train)
    
    # Configure MLPRegressor with improved hyperparameters to prevent model collapse
    # CRITICAL: Use a single MLPRegressor that outputs all time slices simultaneously
    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128),  # Larger architecture for better capacity
        activation='relu',                    # ReLU activation function
        solver='adam',                        # Adam optimizer for efficient training
        alpha=1e-6,                          # Very low regularization to prevent over-regularization
        learning_rate='adaptive',             # Adaptive learning rate
        learning_rate_init=1e-3,             # Initial learning rate
        max_iter=1500,                       # Increased training duration for better convergence
        tol=1e-4,                            # Tolerance for optimization
        random_state=RANDOM_SEED,            # Random state for reproducibility
        early_stopping=True,                 # Enable early stopping
        validation_fraction=0.1,             # Fraction of training data for validation
        n_iter_no_change=20,                 # Patience for early stopping
        shuffle=True,                        # Shuffle samples in each iteration
        batch_size=64,                       # Batch size for training
        beta_1=0.9,                         # Exponential decay rate for first moment estimates
        beta_2=0.999,                       # Exponential decay rate for second moment estimates
        epsilon=1e-8,                       # Value for numerical stability
        verbose=True                        # Enable verbose output to show training progress
    )
    
    # Create Pipeline with StandardScaler for input preprocessing
    # CRITICAL: StandardScaler only applied to inputs, not outputs
    mlp_model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', mlp_regressor)
    ])
    
    # Train the model with progress monitoring
    print(f"Training MLP model (single regressor) on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    print(f"Target shape: {y_train.shape} (predicting {y_train.shape[1]} time slices)")
    print(f"Architecture: {mlp_regressor.hidden_layer_sizes} (total parameters: ~{estimate_mlp_parameters(mlp_regressor.hidden_layer_sizes, X_train.shape[1], y_train.shape[1]):,})")
    print(f"Max iterations: {mlp_regressor.max_iter}, Early stopping patience: {mlp_regressor.n_iter_no_change}")
    print(f"Batch size: {mlp_regressor.batch_size}")
    print("IMPORTANT: Using single MLPRegressor (not MultiOutputRegressor) to prevent model collapse")
    print("This may take several minutes due to the large architecture...")
    print("Progress will be shown below (verbose output from sklearn):")
    print("-" * 60)
    
    import time
    start_time = time.time()
    
    mlp_model.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print("-" * 60)
    print(f"MLP model training completed successfully in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    # Report final training statistics
    mlp_regressor_trained = mlp_model.named_steps['mlp']
    if hasattr(mlp_regressor_trained, 'n_iter_'):
        print(f"Training iterations: {mlp_regressor_trained.n_iter_}")
    
    if hasattr(mlp_regressor_trained, 'loss_'):
        print(f"Final training loss: {mlp_regressor_trained.loss_:.6f}")
    
    if hasattr(mlp_regressor_trained, 'validation_scores_'):
        final_val_score = mlp_regressor_trained.validation_scores_[-1]
        print(f"Final validation score: {final_val_score:.6f}")
    
    return mlp_model


# Bias Correction Module

def compute_bias_corrected_estimator(model, input_data, target_data, ud_indices, bc_indices):
    """
    Compute bias-corrected estimator using the methodology from the research paper.
    
    The bias-corrected estimator is computed as:
    pred_bc[i,t] = mean_over_UD(model(O1[i,src,t])) + mean_over_BC(O2[i,src,t] - model(O1[i,src,t]))
    
    Args:
        model: Trained ML model (GBR or MLP wrapped in MultiOutputRegressor)
        input_data (numpy.ndarray): Input correlator data of shape (N_cfg, 4, N_t)
        target_data (numpy.ndarray): Target correlator data of shape (N_cfg, 4, N_t)
        ud_indices (list): List of (cfg, src) pairs for UD source {3}
        bc_indices (list): List of (cfg, src) pairs for BC source {2}
        
    Returns:
        tuple: (pred_bc, pred_ud)
            - pred_bc: Bias-corrected predictions of shape (N_cfg, N_t)
            - pred_ud: UD-only predictions of shape (N_cfg, N_t)
            
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    n_cfg, n_sources, n_times = input_data.shape
    
    # Validate input dimensions
    if target_data.shape != input_data.shape:
        raise ValueError(f"Input and target data must have same shape. Got {input_data.shape} and {target_data.shape}")
    
    # Validate that we have the expected number of indices
    if len(ud_indices) != n_cfg:
        raise ValueError(f"Expected {n_cfg} UD indices (one per config), got {len(ud_indices)}")
    if len(bc_indices) != n_cfg:
        raise ValueError(f"Expected {n_cfg} BC indices (one per config), got {len(bc_indices)}")
    
    # Initialize output arrays
    pred_bc = np.zeros((n_cfg, n_times))  # Bias-corrected predictions
    pred_ud = np.zeros((n_cfg, n_times))  # UD-only predictions
    
    # Step 1: Compute UD predictions - mean_over_UD(model(O1[i,src,t]))
    # For each configuration, get the UD prediction
    for i, (cfg, src) in enumerate(ud_indices):
        if cfg >= n_cfg or src >= n_sources:
            raise IndexError(f"Invalid UD index ({cfg}, {src}) for data shape ({n_cfg}, {n_sources})")
        
        # Extract input data for this configuration and UD source
        X_ud_single = input_data[cfg, src, :].reshape(1, -1)  # Shape: (1, N_t)
        
        # Get model prediction for UD source
        ud_prediction = model.predict(X_ud_single)  # Shape: (1, N_t)
        pred_ud[cfg, :] = ud_prediction[0, :]  # Store UD prediction
    
    # Step 2: Compute BC corrections - mean_over_BC(O2[i,src,t] - model(O1[i,src,t]))
    bc_corrections = np.zeros((n_cfg, n_times))
    
    for i, (cfg, src) in enumerate(bc_indices):
        if cfg >= n_cfg or src >= n_sources:
            raise IndexError(f"Invalid BC index ({cfg}, {src}) for data shape ({n_cfg}, {n_sources})")
        
        # Extract input and target data for this configuration and BC source
        X_bc_single = input_data[cfg, src, :].reshape(1, -1)   # Shape: (1, N_t)
        y_bc_single = target_data[cfg, src, :]                 # Shape: (N_t,)
        
        # Get model prediction for BC source
        bc_prediction = model.predict(X_bc_single)  # Shape: (1, N_t)
        
        # Compute bias correction: O2[i,src,t] - model(O1[i,src,t])
        bc_corrections[cfg, :] = y_bc_single - bc_prediction[0, :]
    
    # Step 3: Combine UD predictions with BC corrections
    # pred_bc[i,t] = mean_over_UD(model(O1[i,src,t])) + mean_over_BC(O2[i,src,t] - model(O1[i,src,t]))
    # Since we have one UD and one BC source per configuration, the means are just the individual values
    pred_bc = pred_ud + bc_corrections
    
    # Validate output shapes
    if pred_bc.shape != (n_cfg, n_times):
        raise ValueError(f"pred_bc shape mismatch: expected ({n_cfg}, {n_times}), got {pred_bc.shape}")
    if pred_ud.shape != (n_cfg, n_times):
        raise ValueError(f"pred_ud shape mismatch: expected ({n_cfg}, {n_times}), got {pred_ud.shape}")
    
    return pred_bc, pred_ud


# Statistical Analysis Module

def compute_ensemble_statistics(truth_data, gbr_pred_bc, mlp_pred_bc):
    """
    Compute ensemble statistics for correlators including means, standard deviations, and noise-to-signal ratios.
    
    Args:
        truth_data (numpy.ndarray): Truth correlator data of shape (N_cfg, N_t)
        gbr_pred_bc (numpy.ndarray): GBR bias-corrected predictions of shape (N_cfg, N_t)
        mlp_pred_bc (numpy.ndarray): MLP bias-corrected predictions of shape (N_cfg, N_t)
        
    Returns:
        dict: Dictionary containing statistics for each method with structure:
            {
                'truth': {
                    'means': array[N_t],           # μ(τ) for truth
                    'std_devs': array[N_t],        # σ(τ) for truth
                    'nts_ratios': array[N_t]       # σ(τ)/|μ(τ)| for truth
                },
                'gbr': {
                    'means': array[N_t],           # μ(τ) for GBR bias-corrected
                    'std_devs': array[N_t],        # σ(τ) for GBR bias-corrected  
                    'nts_ratios': array[N_t]       # σ(τ)/|μ(τ)| for GBR bias-corrected
                },
                'mlp': {
                    'means': array[N_t],           # μ(τ) for MLP bias-corrected
                    'std_devs': array[N_t],        # σ(τ) for MLP bias-corrected
                    'nts_ratios': array[N_t]       # σ(τ)/|μ(τ)| for MLP bias-corrected
                }
            }
            
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    # Validate inputs for statistical computation
    data_arrays = [truth_data, gbr_pred_bc, mlp_pred_bc]
    method_names = ['truth', 'gbr', 'mlp']
    validate_statistical_computation_inputs(data_arrays, method_names)
    
    n_cfg, n_times = truth_data.shape
    
    # Initialize results dictionary
    statistics = {
        'truth': {'means': None, 'std_devs': None, 'nts_ratios': None},
        'gbr': {'means': None, 'std_devs': None, 'nts_ratios': None},
        'mlp': {'means': None, 'std_devs': None, 'nts_ratios': None}
    }
    
    # Process each dataset
    datasets = {
        'truth': truth_data,
        'gbr': gbr_pred_bc,
        'mlp': mlp_pred_bc
    }
    
    for method_name, data in datasets.items():
        # Compute ensemble means μ(τ) = mean over configurations (axis=0)
        # Requirements: 5.1
        ensemble_means = np.mean(data, axis=0)  # Shape: (N_t,)
        
        # Compute ensemble standard deviations σ(τ) = std over configurations (axis=0)
        # Requirements: 5.2
        ensemble_std_devs = np.std(data, axis=0, ddof=1)  # Shape: (N_t,), using sample std (ddof=1)
        
        # Compute noise-to-signal ratios NtS(τ) = σ(τ) / |μ(τ)|
        # Requirements: 5.3, 5.5 (handle division by zero)
        nts_ratios = np.zeros_like(ensemble_means)  # Initialize with zeros
        
        # Handle division by zero in NtS calculations
        # Use a small epsilon to avoid division by exactly zero
        epsilon = 1e-15  # Small value to prevent division by zero
        abs_means = np.abs(ensemble_means)
        
        # Only compute NtS where |μ(τ)| > epsilon
        valid_mask = abs_means > epsilon
        nts_ratios[valid_mask] = ensemble_std_devs[valid_mask] / abs_means[valid_mask]
        
        # For points where |μ(τ)| ≤ epsilon, set NtS to infinity (or a large value)
        nts_ratios[~valid_mask] = np.inf
        
        # Store results
        statistics[method_name]['means'] = ensemble_means
        statistics[method_name]['std_devs'] = ensemble_std_devs
        statistics[method_name]['nts_ratios'] = nts_ratios
        
        # Validate output shapes
        if ensemble_means.shape != (n_times,):
            raise ValueError(f"{method_name} ensemble_means shape mismatch: expected ({n_times},), got {ensemble_means.shape}")
        if ensemble_std_devs.shape != (n_times,):
            raise ValueError(f"{method_name} ensemble_std_devs shape mismatch: expected ({n_times},), got {ensemble_std_devs.shape}")
        if nts_ratios.shape != (n_times,):
            raise ValueError(f"{method_name} nts_ratios shape mismatch: expected ({n_times},), got {nts_ratios.shape}")
    
    # Requirements: 5.4 - Process multiple estimators (Truth, GBR bias-corrected, and MLP bias-corrected)
    print(f"Computed ensemble statistics for {n_cfg} configurations and {n_times} time slices")
    print(f"Methods processed: Truth, GBR bias-corrected, MLP bias-corrected")
    
    return statistics


# Spectral Fitting Module

def multi_exponential_correlator(tau, params, T=96):
    """
    Multi-exponential correlator function following Eq. (4) from the paper:
    C(τ) = Σ_{n=0}^{N_states-1} (-1)^{n(τ+1)} * (a_n^2 / (2 E_n)) * (e^{-E_n τ} + e^{-E_n (T−τ)})
    
    Args:
        tau (array): Time values
        params (array): Flattened parameters [a0, E0, a1, E1, ...]
        T (int): Temporal extent
    
    Returns:
        array: Correlator values
    """
    tau = np.asarray(tau, dtype=float)
    params = np.asarray(params, dtype=float)
    n_states = len(params) // 2
    result = np.zeros_like(tau, dtype=float)
    
    for n in range(n_states):
        a_n = float(params[2*n])
        E_n = float(params[2*n + 1])
        
        # Avoid numerical issues with very small energies
        if E_n <= 0:
            continue
            
        sign = (-1)**(n * (tau + 1))
        amplitude = a_n**2 / (2 * E_n)
        exponential = np.exp(-E_n * tau) + np.exp(-E_n * (T - tau))
        
        result += sign * amplitude * exponential
    
    return result


def fit_spectral_parameters(time_values, correlator_mean, correlator_cov=None, n_states=2, t_min=3, t_max=40, T=96):
    """
    Perform multi-exponential fits for correlator data using scipy.optimize.curve_fit.
    
    Args:
        time_values (array): Time slice values
        correlator_mean (array): Ensemble-average correlator
        correlator_cov (array): Covariance matrix or diagonal variances (optional)
        n_states (int): Number of states to fit (default: 2)
        t_min (int): Minimum time for fitting (default: 3)
        t_max (int): Maximum time for fitting (default: 40)
        T (int): Temporal extent (default: 96)
    
    Returns:
        dict: Fit results with parameters, errors, chi2/dof, etc.
    """
    from scipy.optimize import curve_fit
    
    # Select fitting range
    fit_mask = (time_values >= t_min) & (time_values <= t_max)
    tau_fit = time_values[fit_mask]
    data_fit = correlator_mean[fit_mask]
    
    if len(tau_fit) < 2 * n_states:
        return {
            'success': False,
            'error': f'Not enough data points for fit: {len(tau_fit)} < {2*n_states}'
        }
    
    # Define fitting function
    def fit_function(tau, *params):
        return multi_exponential_correlator(tau, params, T)
    
    # Initial guesses and bounds
    p0 = []
    bounds_lower = []
    bounds_upper = []
    
    for n in range(n_states):
        # Amplitude initial guess and bounds
        p0.append(1.0)
        bounds_lower.append(0.1)
        bounds_upper.append(10.0)
        
        # Energy initial guess and bounds (ordered: E0 < E1 < E2 ...)
        E_guess = 0.3 + n * 0.4
        p0.append(E_guess)
        bounds_lower.append(0.05 + n * 0.1)  # Ensure ordering
        bounds_upper.append(3.0)
    
    bounds = (bounds_lower, bounds_upper)
    
    # Estimate uncertainties
    if correlator_cov is not None:
        if correlator_cov.ndim == 2:
            # Full covariance matrix
            cov_fit = correlator_cov[np.ix_(fit_mask, fit_mask)]
            try:
                sigma = np.sqrt(np.diag(cov_fit))
            except:
                sigma = np.abs(data_fit) * 0.05 + 1e-8
        else:
            # Diagonal variances
            sigma = np.sqrt(correlator_cov[fit_mask])
    else:
        # Estimate as 5% of data + small constant
        sigma = np.abs(data_fit) * 0.05 + 1e-8
    
    try:
        # Perform fit
        popt, pcov = curve_fit(
            fit_function, tau_fit, data_fit, 
            p0=p0, bounds=bounds, sigma=sigma,
            maxfev=5000, method='trf'
        )
        
        # Extract parameter errors
        param_errors = np.sqrt(np.diag(pcov))
        
        # Compute chi2/dof
        y_pred = fit_function(tau_fit, *popt)
        residuals = (data_fit - y_pred) / sigma
        chi2 = np.sum(residuals**2)
        dof = len(tau_fit) - len(popt)
        chi2_dof = chi2 / max(dof, 1)
        
        # Compute p-value (rough approximation)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, dof) if dof > 0 else 0.5
        
        # Build results dictionary
        results = {
            'success': True,
            'method': 'scipy.curve_fit',
            'chi2_dof': chi2_dof,
            'p_value': p_value,
            'dof': dof,
            'fit_range': f't_min={t_min}, t_max={t_max}',
            'n_states': n_states
        }
        
        # Extract individual parameters
        for n in range(min(2, n_states)):  # Only report first 2 states
            a_idx = 2 * n
            E_idx = 2 * n + 1
            
            results[f'a{n}'] = popt[a_idx]
            results[f'a{n}_err'] = param_errors[a_idx]
            results[f'E{n}'] = popt[E_idx]
            results[f'E{n}_err'] = param_errors[E_idx]
            
            # Compute energy differences
            if n == 0:
                results[f'dE{n}'] = popt[E_idx]  # dE0 = E0
                results[f'dE{n}_err'] = param_errors[E_idx]
            else:
                results[f'dE{n}'] = popt[E_idx] - popt[0]  # dE1 = E1 - E0
                # Error propagation for difference
                results[f'dE{n}_err'] = np.sqrt(param_errors[E_idx]**2 + param_errors[0]**2)
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'method': 'scipy.curve_fit',
            'error': f'Fit failed: {str(e)}'
        }


# Extended Visualization Module

def plot_bias_correction_effect(time_values, C_truth, C_pred_uncorr, C_pred_corr, method_name="GBR"):
    """
    Plot the bias correction effect showing relative correlated difference vs time.
    Reproduces Figure 1 (right) style from the LATTICE 2024 paper.
    
    IMPROVED VERSION: Applies magnitude masking and restricts plotting range to avoid spikes.
    
    Args:
        time_values (array): Time slice values
        C_truth (array): Truth ensemble means
        C_pred_uncorr (array): Uncorrected predictions ensemble means
        C_pred_corr (array): Bias-corrected predictions ensemble means
        method_name (str): Name of the ML method
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Apply truth magnitude mask to ignore time slices with very small truth values
    abs_truth = np.abs(C_truth)
    magnitude_mask = abs_truth > TRUTH_MAGNITUDE_THRESHOLD
    
    # Apply time range mask
    time_range_mask = (time_values >= TAU_MIN) & (time_values <= TAU_MAX)
    
    # Combine both masks
    valid_mask = magnitude_mask & time_range_mask
    
    if not np.any(valid_mask):
        print(f"Warning: No valid data points for {method_name} bias correction plot")
        ax.text(0.5, 0.5, f'No valid data\n(all |C_truth| < {TRUTH_MAGNITUDE_THRESHOLD:.0e})', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_xlabel('Euclidean Time τ', fontsize=14)
        ax.set_ylabel('Relative Difference Δ(τ)', fontsize=14)
        ax.set_title(f'Bias Correction Effect - {method_name}', fontsize=16, pad=20)
        plt.tight_layout()
        return fig
    
    # Compute relative differences only for valid points
    tau_valid = time_values[valid_mask]
    truth_valid = C_truth[valid_mask]
    uncorr_valid = C_pred_uncorr[valid_mask]
    corr_valid = C_pred_corr[valid_mask]
    
    # Compute relative differences: Δ(τ) = (C_pred(τ) − C_truth(τ)) / |C_truth(τ)|
    delta_uncorr = (uncorr_valid - truth_valid) / np.abs(truth_valid)
    delta_corr = (corr_valid - truth_valid) / np.abs(truth_valid)
    
    # Plot both curves
    ax.plot(tau_valid, delta_uncorr, 'o--', 
            label=f'{method_name} Uncorrected', linewidth=2, markersize=5, color='red', alpha=0.8)
    ax.plot(tau_valid, delta_corr, 's-', 
            label=f'{method_name} Bias-Corrected', linewidth=2, markersize=5, color='blue', alpha=0.8)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Set fixed axis limits for readability
    ax.set_xlim(TAU_MIN - 2, TAU_MAX + 2)
    ax.set_ylim(BIAS_PLOT_Y_LIMITS)
    
    ax.set_xlabel('Euclidean Time τ', fontsize=14)
    ax.set_ylabel('Relative Difference Δ(τ)', fontsize=14)
    ax.set_title(f'Bias Correction Effect - {method_name}', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    # Add text annotation about the filtering
    ax.text(0.02, 0.98, f'τ ∈ [{TAU_MIN}, {TAU_MAX}], |C_truth| > {TRUTH_MAGNITUDE_THRESHOLD:.0e}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_full_correlator_comparison(time_values, truth_all, truth_train, pred_uncorr, pred_corr, method_name="GBR"):
    """
    Plot correlator comparison with multiple curves (Figure 2 left style).
    
    Args:
        time_values (array): Time slice values
        truth_all (array): Truth ensemble mean (all sources)
        truth_train (array): Truth ensemble mean (training sources only)
        pred_uncorr (array): Uncorrected predictions ensemble mean
        pred_corr (array): Bias-corrected predictions ensemble mean
        method_name (str): Name of the ML method
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all curves on log scale
    ax.semilogy(time_values, np.abs(truth_all), 'o-', 
                label='Truth (All Sources)', linewidth=2, markersize=6, color='black')
    ax.semilogy(time_values, np.abs(truth_train), 'd-', 
                label='Truth (Training Only)', linewidth=2, markersize=5, color='gray', alpha=0.8)
    ax.semilogy(time_values, np.abs(pred_uncorr), 's--', 
                label=f'{method_name} Uncorrected', linewidth=2, markersize=5, color='red', alpha=0.8)
    ax.semilogy(time_values, np.abs(pred_corr), '^-', 
                label=f'{method_name} Bias-Corrected', linewidth=2, markersize=5, color='blue', alpha=0.8)
    
    ax.set_xlabel('Euclidean Time τ', fontsize=14)
    ax.set_ylabel('|Correlator|', fontsize=14)
    ax.set_title(f'Correlator Comparison - {method_name}', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_full_nts_comparison(time_values, nts_truth_all, nts_truth_train, nts_uncorr, nts_corr, method_name="GBR"):
    """
    Plot NtS comparison with multiple curves (Figure 2 right style).
    
    Args:
        time_values (array): Time slice values
        nts_truth_all (array): Truth NtS (all sources)
        nts_truth_train (array): Truth NtS (training sources only)
        nts_uncorr (array): Uncorrected predictions NtS
        nts_corr (array): Bias-corrected predictions NtS
        method_name (str): Name of the ML method
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Handle infinite values for plotting
    def prepare_for_plot(nts_array):
        finite_vals = nts_array[np.isfinite(nts_array)]
        if len(finite_vals) > 0:
            max_finite = np.max(finite_vals)
            inf_replacement = max_finite * 10
        else:
            inf_replacement = 10.0
        return np.where(np.isfinite(nts_array), nts_array, inf_replacement)
    
    nts_truth_all_plot = prepare_for_plot(nts_truth_all)
    nts_truth_train_plot = prepare_for_plot(nts_truth_train)
    nts_uncorr_plot = prepare_for_plot(nts_uncorr)
    nts_corr_plot = prepare_for_plot(nts_corr)
    
    # Plot all curves on log scale
    ax.semilogy(time_values, nts_truth_all_plot, 'o-', 
                label='Truth (All Sources)', linewidth=2, markersize=6, color='black')
    ax.semilogy(time_values, nts_truth_train_plot, 'd-', 
                label='Truth (Training Only)', linewidth=2, markersize=5, color='gray', alpha=0.8)
    ax.semilogy(time_values, nts_uncorr_plot, 's--', 
                label=f'{method_name} Uncorrected', linewidth=2, markersize=5, color='red', alpha=0.8)
    ax.semilogy(time_values, nts_corr_plot, '^-', 
                label=f'{method_name} Bias-Corrected', linewidth=2, markersize=5, color='blue', alpha=0.8)
    
    ax.set_xlabel('Euclidean Time τ', fontsize=14)
    ax.set_ylabel('Noise-to-Signal Ratio', fontsize=14)
    ax.set_title(f'NtS Comparison - {method_name}', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_fit_parameter_comparison(fit_results_dict):
    """
    Plot spectral fit parameters comparison (Figure 3 style).
    
    Args:
        fit_results_dict (dict): Dictionary with fit results for each method
                               {'truth': {...}, 'gbr': {...}, 'mlp': {...}}ruth': fit_result, 'gbr': fit_result, 'mlp': fit_result}
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    methods = ['truth', 'gbr', 'mlp']
    method_labels = ['Truth', 'GBR', 'MLP']
    colors = ['black', 'red', 'blue']
    
    parameters = ['a0', 'a1', 'dE0', 'dE1']
    param_labels = ['$a_0$', '$a_1$', '$\\Delta E_0$', '$\\Delta E_1$']
    
    for i, (param, param_label) in enumerate(zip(parameters, param_labels)):
        ax = axes[i]
        
        # Get truth values for reference band
        if fit_results_dict['truth']['success'] and param in fit_results_dict['truth']:
            truth_val = fit_results_dict['truth'][param]
            truth_err = fit_results_dict['truth'].get(f'{param}_err', 0)
            
            # Plot truth as horizontal band
            ax.axhspan(truth_val - truth_err, truth_val + truth_err, 
                      alpha=0.2, color='gray', label='Truth ±1σ')
            ax.axhline(truth_val, color='black', linestyle='-', alpha=0.5)
        
        # Plot method values as points with error bars
        x_positions = []
        values = []
        errors = []
        labels = []
        
        for j, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
            if method == 'truth':
                continue  # Already plotted as band
                
            if (fit_results_dict[method]['success'] and 
                param in fit_results_dict[method]):
                
                x_positions.append(j)
                values.append(fit_results_dict[method][param])
                errors.append(fit_results_dict[method].get(f'{param}_err', 0))
                labels.append(label)
        
        if x_positions:
            ax.errorbar(x_positions, values, yerr=errors, 
                       fmt='o', markersize=8, capsize=5, capthick=2,
                       color='red' if len(x_positions) == 1 else None)
        
        ax.set_ylabel(param_label, fontsize=14)
        ax.set_xticks(range(len(method_labels)))
        ax.set_xticklabels(method_labels)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='best', fontsize=10)
    
    plt.suptitle('Spectral Fit Parameters Comparison', fontsize=16)
    plt.tight_layout()
    return fig


def print_fit_parameters_table(fit_results_dict):
    """
    Print a formatted table of fit parameters similar to Table 3 in the paper.
    
    Args:
        fit_results_dict (dict): Dictionary with fit results for each method
    """
    print("\n" + "="*80)
    print("SPECTRAL FIT PARAMETERS TABLE")
    print("="*80)
    print(f"{'Method':<10} {'a0':<15} {'a1':<15} {'dE0':<15} {'dE1':<15} {'χ²/dof':<10} {'Q':<8}")
    print("-"*80)
    
    for method in ['truth', 'gbr', 'mlp']:
        method_name = method.upper()
        result = fit_results_dict[method]
        
        if result['success']:
            # Format parameters with errors
            params_str = []
            for param in ['a0', 'a1', 'dE0', 'dE1']:
                if param in result:
                    val = result[param]
                    err = result.get(f'{param}_err', 0)
                    params_str.append(f"{val:.4f}({err:.4f})")
                else:
                    params_str.append("N/A")
            
            chi2_str = f"{result.get('chi2_dof', 0):.2f}"
            q_str = f"{result.get('p_value', 0):.3f}"
            
            print(f"{method_name:<10} {params_str[0]:<15} {params_str[1]:<15} {params_str[2]:<15} {params_str[3]:<15} {chi2_str:<10} {q_str:<8}")
        else:
            print(f"{method_name:<10} {'FIT FAILED':<60}")
    
    print("="*80)


# Original Visualization Module

def plot_correlators(time_values, truth_target, pred_bc_gbr, pred_bc_mlp):
    """
    Generate log-scale plots of correlator magnitude vs Euclidean time.
    
    Creates comparative plots showing Truth, GBR bias-corrected, and MLP bias-corrected
    correlator curves with proper scientific styling and labels.
    
    Args:
        time_values (numpy.ndarray): Array of time slice indices [0, 1, 2, ..., N_t-1]
        truth_target (numpy.ndarray): Truth correlator ensemble means of shape (N_t,)
        pred_bc_gbr (numpy.ndarray): GBR bias-corrected ensemble means of shape (N_t,)
        pred_bc_mlp (numpy.ndarray): MLP bias-corrected ensemble means of shape (N_t,)
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the correlator plot
        
    Requirements: 6.1, 6.3, 6.5
    """
    # Validate input shapes
    if not isinstance(time_values, np.ndarray) or len(time_values.shape) != 1:
        raise ValueError(f"time_values must be a 1D numpy array, got shape {time_values.shape if hasattr(time_values, 'shape') else type(time_values)}")
    
    n_times = len(time_values)
    
    for name, data in [('truth_target', truth_target), ('pred_bc_gbr', pred_bc_gbr), ('pred_bc_mlp', pred_bc_mlp)]:
        if not isinstance(data, np.ndarray) or len(data.shape) != 1:
            raise ValueError(f"{name} must be a 1D numpy array, got shape {data.shape if hasattr(data, 'shape') else type(data)}")
        if len(data) != n_times:
            raise ValueError(f"{name} length ({len(data)}) must match time_values length ({n_times})")
    
    # Create figure with scientific plot styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot correlator magnitude vs Euclidean time on log scale
    # Use absolute values to handle potential negative correlators in log scale
    ax.semilogy(time_values, np.abs(truth_target), 'o-', 
                label='Truth', linewidth=2, markersize=6, color='black')
    ax.semilogy(time_values, np.abs(pred_bc_gbr), 's--', 
                label='GBR Bias-Corrected', linewidth=2, markersize=5, color='red', alpha=0.8)
    ax.semilogy(time_values, np.abs(pred_bc_mlp), '^:', 
                label='MLP Bias-Corrected', linewidth=2, markersize=5, color='blue', alpha=0.8)
    
    # Apply proper scientific plot styling and labels
    ax.set_xlabel('Euclidean Time τ', fontsize=14)
    ax.set_ylabel('|Correlator|', fontsize=14)
    ax.set_title('Lattice QCD 2-Point Correlator Analysis', fontsize=16, pad=20)
    
    # Configure grid and legend
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    # Set axis limits and ticks for better visualization
    ax.set_xlim(time_values[0] - 0.5, time_values[-1] + 0.5)
    ax.set_xticks(time_values[::max(1, len(time_values)//10)])  # Show every ~10th tick for readability
    
    # Ensure y-axis shows appropriate range for correlator data
    y_min = min(np.min(np.abs(truth_target)), np.min(np.abs(pred_bc_gbr)), np.min(np.abs(pred_bc_mlp)))
    y_max = max(np.max(np.abs(truth_target)), np.max(np.abs(pred_bc_gbr)), np.max(np.abs(pred_bc_mlp)))
    
    # Add some padding to y-axis limits
    y_min_padded = y_min * 0.5 if y_min > 0 else y_min * 1.5
    y_max_padded = y_max * 2.0
    ax.set_ylim(y_min_padded, y_max_padded)
    
    # Improve layout
    plt.tight_layout()
    
    return fig


def plot_noise_to_signal(time_values, truth_nts, gbr_nts, mlp_nts):
    """
    Generate log-scale plots of noise-to-signal ratio vs Euclidean time.
    
    Creates comparative plots showing NtS ratios for Truth, GBR bias-corrected, 
    and MLP bias-corrected estimators with proper legends and scientific styling.
    
    Args:
        time_values (numpy.ndarray): Array of time slice indices [0, 1, 2, ..., N_t-1]
        truth_nts (numpy.ndarray): Truth NtS ratios of shape (N_t,)
        gbr_nts (numpy.ndarray): GBR bias-corrected NtS ratios of shape (N_t,)
        mlp_nts (numpy.ndarray): MLP bias-corrected NtS ratios of shape (N_t,)
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the noise-to-signal plot
        
    Requirements: 6.2, 6.3, 6.5
    """
    # Validate input shapes
    if not isinstance(time_values, np.ndarray) or len(time_values.shape) != 1:
        raise ValueError(f"time_values must be a 1D numpy array, got shape {time_values.shape if hasattr(time_values, 'shape') else type(time_values)}")
    
    n_times = len(time_values)
    
    for name, data in [('truth_nts', truth_nts), ('gbr_nts', gbr_nts), ('mlp_nts', mlp_nts)]:
        if not isinstance(data, np.ndarray) or len(data.shape) != 1:
            raise ValueError(f"{name} must be a 1D numpy array, got shape {data.shape if hasattr(data, 'shape') else type(data)}")
        if len(data) != n_times:
            raise ValueError(f"{name} length ({len(data)}) must match time_values length ({n_times})")
    
    # Create figure with scientific plot styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Handle infinite values in NtS ratios for plotting
    # Replace infinite values with a large finite value for visualization
    max_finite_truth = np.max(truth_nts[np.isfinite(truth_nts)]) if np.any(np.isfinite(truth_nts)) else 1.0
    max_finite_gbr = np.max(gbr_nts[np.isfinite(gbr_nts)]) if np.any(np.isfinite(gbr_nts)) else 1.0
    max_finite_mlp = np.max(mlp_nts[np.isfinite(mlp_nts)]) if np.any(np.isfinite(mlp_nts)) else 1.0
    
    # Use 10x the maximum finite value as replacement for infinite values
    inf_replacement = 10 * max(max_finite_truth, max_finite_gbr, max_finite_mlp, 1.0)
    
    # Create copies with infinite values replaced
    truth_nts_plot = np.where(np.isfinite(truth_nts), truth_nts, inf_replacement)
    gbr_nts_plot = np.where(np.isfinite(gbr_nts), gbr_nts, inf_replacement)
    mlp_nts_plot = np.where(np.isfinite(mlp_nts), mlp_nts, inf_replacement)
    
    # Plot NtS ratios vs Euclidean time on log scale
    ax.semilogy(time_values, truth_nts_plot, 'o-', 
                label='Truth', linewidth=2, markersize=6, color='black')
    ax.semilogy(time_values, gbr_nts_plot, 's--', 
                label='GBR Bias-Corrected', linewidth=2, markersize=5, color='red', alpha=0.8)
    ax.semilogy(time_values, mlp_nts_plot, '^:', 
                label='MLP Bias-Corrected', linewidth=2, markersize=5, color='blue', alpha=0.8)
    
    # Apply proper scientific plot styling and labels
    ax.set_xlabel('Euclidean Time τ', fontsize=14)
    ax.set_ylabel('Noise-to-Signal Ratio', fontsize=14)
    ax.set_title('Noise-to-Signal Ratio Comparison', fontsize=16, pad=20)
    
    # Configure grid and legend
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    # Set axis limits and ticks for better visualization
    ax.set_xlim(time_values[0] - 0.5, time_values[-1] + 0.5)
    ax.set_xticks(time_values[::max(1, len(time_values)//10)])  # Show every ~10th tick for readability
    
    # Set y-axis limits based on finite values
    finite_values = np.concatenate([
        truth_nts_plot[np.isfinite(truth_nts)],
        gbr_nts_plot[np.isfinite(gbr_nts)],
        mlp_nts_plot[np.isfinite(mlp_nts)]
    ])
    
    if len(finite_values) > 0:
        y_min = np.min(finite_values)
        y_max = np.max(finite_values)
        
        # Add padding to y-axis limits
        y_min_padded = y_min * 0.5 if y_min > 0 else y_min * 1.5
        y_max_padded = y_max * 2.0
        ax.set_ylim(y_min_padded, y_max_padded)
    else:
        # Fallback if all values are infinite
        ax.set_ylim(0.1, 10.0)
    
    # Improve layout
    plt.tight_layout()
    
    return fig


def main():
    """
    Main analysis function that orchestrates the complete lattice QCD experiment.
    
    Implements the complete workflow:
    - Choose experiment configuration
    - Load correlator data from CSV files
    - Train GBR and MLP models
    - Compute bias-corrected estimators
    - Generate ensemble statistics
    - Create comparative plots
    
    Requirements: 7.2
    """
    print("Lattice QCD Analysis Pipeline")
    print("=" * 40)
    print(f"Random seed set to: {RANDOM_SEED}")
    print(f"Time source partitioning:")
    print(f"  TRAIN sources: {TRAIN_SOURCES}")
    print(f"  BC source: {BC_SOURCE}")
    print(f"  UD source: {UD_SOURCE}")
    print("=" * 40)
    
    try:
        # Step 0: Choose experiment configuration
        print("\n0. Selecting experiment configuration...")
        experiment_cfg = choose_experiment()
        experiment_label = experiment_cfg["label"]
        input_path = experiment_cfg["input_csv"]
        target_path = experiment_cfg["target_csv"]
        
        # Step 1: Load data using selected CSV files
        print("\n1. Loading correlator data...")
        
        input_data, target_data, truth_input, truth_target, time_values = load_correlator_data(
            input_path, target_path
        )
        
        print(f"   Input data shape: {input_data.shape}")
        print(f"   Target data shape: {target_data.shape}")
        print(f"   Truth input value: {truth_input}")
        print(f"   Truth target value: {truth_target}")
        print(f"   Time values: {len(time_values)} time slices")
        
        # Step 2: Reshape data and create partitions
        print("\n2. Preprocessing data...")
        n_rows, n_times = input_data.shape
        n_configs = n_rows // N_TIME_SOURCES
        
        print(f"   Number of configurations: {n_configs}")
        print(f"   Number of time sources per config: {N_TIME_SOURCES}")
        print(f"   Number of time slices: {n_times}")
        
        # Reshape correlator data
        input_reshaped = reshape_correlator_data(input_data, n_configs, N_TIME_SOURCES, n_times)
        target_reshaped = reshape_correlator_data(target_data, n_configs, N_TIME_SOURCES, n_times)
        
        print(f"   Reshaped input data: {input_reshaped.shape}")
        print(f"   Reshaped target data: {target_reshaped.shape}")
        
        # Create time source partitions
        train_indices, bc_indices, ud_indices = create_time_source_partitions(n_configs)
        
        print(f"   TRAIN partition size: {len(train_indices)} samples")
        print(f"   BC partition size: {len(bc_indices)} samples")
        print(f"   UD partition size: {len(ud_indices)} samples")
        
        # Prepare ML datasets
        X_train, y_train, X_bc, y_bc, X_ud, y_ud = prepare_ml_datasets(
            input_reshaped, target_reshaped, train_indices, bc_indices, ud_indices
        )
        
        print(f"   Training dataset: X_train {X_train.shape}, y_train {y_train.shape}")
        
        # Step 3: Train both GBR and MLP models
        print("\n3. Training machine learning models...")
        
        # Train GBR model (using original data - GBR doesn't need normalization)
        print("   Training GBR model...")
        gbr_model = train_gbr_model(X_train, y_train)
        
        # Train MLP model (Pipeline with StandardScaler handles normalization automatically)
        print("   Training MLP model...")
        mlp_model = train_mlp_model(X_train, y_train)
        
        print("   Both models trained successfully")
        
        # Step 4: Compute bias-corrected estimators
        print("\n4. Computing bias-corrected estimators...")
        
        # Compute GBR bias-corrected estimator (using original input data)
        print("   Computing GBR bias-corrected predictions...")
        gbr_pred_bc, gbr_pred_ud = compute_bias_corrected_estimator(
            gbr_model, input_reshaped, target_reshaped, ud_indices, bc_indices
        )
        
        # Compute MLP bias-corrected estimator (Pipeline handles normalization automatically)
        print("   Computing MLP bias-corrected predictions...")
        mlp_pred_bc, mlp_pred_ud = compute_bias_corrected_estimator(
            mlp_model, input_reshaped, target_reshaped, ud_indices, bc_indices
        )
        
        print(f"   GBR bias-corrected predictions: {gbr_pred_bc.shape}")
        print(f"   MLP bias-corrected predictions: {mlp_pred_bc.shape}")
        
        # Prepare truth data for ensemble statistics (average over all time sources)
        truth_data = np.mean(target_reshaped, axis=1)  # Average over time sources, shape: (N_cfg, N_t)
        
        print(f"   Truth data for statistics: {truth_data.shape}")
        
        # Step 5: Generate ensemble statistics
        print("\n5. Computing ensemble statistics...")
        
        statistics = compute_ensemble_statistics(truth_data, gbr_pred_bc, mlp_pred_bc)
        
        # Print summary statistics
        print(f"   Ensemble statistics computed for experiment: {experiment_label}")
        for method in ['truth', 'gbr', 'mlp']:
            means = statistics[method]['means']
            std_devs = statistics[method]['std_devs']
            nts_ratios = statistics[method]['nts_ratios']
            
            # Compute average NtS ratio (excluding infinite values)
            finite_nts = nts_ratios[np.isfinite(nts_ratios)]
            avg_nts = np.mean(finite_nts) if len(finite_nts) > 0 else np.inf
            
            print(f"     {method.upper()}: mean correlator range [{np.min(np.abs(means)):.2e}, {np.max(np.abs(means)):.2e}]")
            print(f"              average NtS ratio: {avg_nts:.3f}")
        
        # Step 6: Compute additional statistics for extended plots
        print("\n6. Computing extended statistics...")
        
        # Compute training-only truth statistics
        truth_train_data = np.zeros((n_configs, n_times))
        for cfg in range(n_configs):
            # Average over training sources only (sources 0 and 1)
            truth_train_data[cfg, :] = np.mean(target_reshaped[cfg, TRAIN_SOURCES, :], axis=0)
        
        # Compute statistics for training-only truth
        truth_train_means = np.mean(truth_train_data, axis=0)
        truth_train_std = np.std(truth_train_data, axis=0, ddof=1)
        
        # Compute NtS for training-only truth
        epsilon = 1e-15
        abs_means_train = np.abs(truth_train_means)
        valid_mask_train = abs_means_train > epsilon
        truth_train_nts = np.zeros_like(truth_train_means)
        truth_train_nts[valid_mask_train] = truth_train_std[valid_mask_train] / abs_means_train[valid_mask_train]
        truth_train_nts[~valid_mask_train] = np.inf
        
        # Compute statistics for uncorrected predictions
        gbr_uncorr_means = np.mean(gbr_pred_ud, axis=0)
        gbr_uncorr_std = np.std(gbr_pred_ud, axis=0, ddof=1)
        abs_means_gbr_uncorr = np.abs(gbr_uncorr_means)
        valid_mask_gbr_uncorr = abs_means_gbr_uncorr > epsilon
        gbr_uncorr_nts = np.zeros_like(gbr_uncorr_means)
        gbr_uncorr_nts[valid_mask_gbr_uncorr] = gbr_uncorr_std[valid_mask_gbr_uncorr] / abs_means_gbr_uncorr[valid_mask_gbr_uncorr]
        gbr_uncorr_nts[~valid_mask_gbr_uncorr] = np.inf
        
        mlp_uncorr_means = np.mean(mlp_pred_ud, axis=0)
        mlp_uncorr_std = np.std(mlp_pred_ud, axis=0, ddof=1)
        abs_means_mlp_uncorr = np.abs(mlp_uncorr_means)
        valid_mask_mlp_uncorr = abs_means_mlp_uncorr > epsilon
        mlp_uncorr_nts = np.zeros_like(mlp_uncorr_means)
        mlp_uncorr_nts[valid_mask_mlp_uncorr] = mlp_uncorr_std[valid_mask_mlp_uncorr] / abs_means_mlp_uncorr[valid_mask_mlp_uncorr]
        mlp_uncorr_nts[~valid_mask_mlp_uncorr] = np.inf
        
        print("   Extended statistics computed")
        
        # Step 7: Perform spectral fits
        print("\n7. Performing spectral fits...")
        
        fit_results = {}
        
        # Fit truth data (all sources)
        print("   Fitting truth correlator...")
        try:
            fit_results['truth'] = fit_spectral_parameters(
                time_values, statistics['truth']['means'], 
                n_states=2, t_min=3, t_max=40
            )
            if fit_results['truth']['success']:
                print(f"     Truth fit successful: χ²/dof = {fit_results['truth']['chi2_dof']:.3f}")
            else:
                print(f"     Truth fit failed: {fit_results['truth']['error']}")
        except Exception as e:
            fit_results['truth'] = {'success': False, 'error': f'Exception: {str(e)}'}
            print(f"     Truth fit exception: {str(e)}")
        
        # Fit GBR bias-corrected data
        print("   Fitting GBR bias-corrected correlator...")
        try:
            fit_results['gbr'] = fit_spectral_parameters(
                time_values, statistics['gbr']['means'], 
                n_states=2, t_min=3, t_max=40
            )
            if fit_results['gbr']['success']:
                print(f"     GBR fit successful: χ²/dof = {fit_results['gbr']['chi2_dof']:.3f}")
            else:
                print(f"     GBR fit failed: {fit_results['gbr']['error']}")
        except Exception as e:
            fit_results['gbr'] = {'success': False, 'error': f'Exception: {str(e)}'}
            print(f"     GBR fit exception: {str(e)}")
        
        # Fit MLP bias-corrected data
        print("   Fitting MLP bias-corrected correlator...")
        try:
            fit_results['mlp'] = fit_spectral_parameters(
                time_values, statistics['mlp']['means'], 
                n_states=2, t_min=3, t_max=40
            )
            if fit_results['mlp']['success']:
                print(f"     MLP fit successful: χ²/dof = {fit_results['mlp']['chi2_dof']:.3f}")
            else:
                print(f"     MLP fit failed: {fit_results['mlp']['error']}")
        except Exception as e:
            fit_results['mlp'] = {'success': False, 'error': f'Exception: {str(e)}'}
            print(f"     MLP fit exception: {str(e)}")
        
        # Print fit results table
        print(f"\n=== Summary for experiment: {experiment_label} ===")
        print_fit_parameters_table(fit_results)
        
        # Step 8: Create all plots
        print("\n8. Generating all visualization plots...")
        
        # Original plots
        correlator_fig = plot_correlators(
            time_values, 
            statistics['truth']['means'],
            statistics['gbr']['means'], 
            statistics['mlp']['means']
        )
        
        nts_fig = plot_noise_to_signal(
            time_values,
            statistics['truth']['nts_ratios'],
            statistics['gbr']['nts_ratios'],
            statistics['mlp']['nts_ratios']
        )
        
        # New plots - Figure 1 style (bias correction effect)
        bias_correction_gbr_fig = plot_bias_correction_effect(
            time_values, statistics['truth']['means'], 
            gbr_uncorr_means, statistics['gbr']['means'], "GBR"
        )
        
        bias_correction_mlp_fig = plot_bias_correction_effect(
            time_values, statistics['truth']['means'], 
            mlp_uncorr_means, statistics['mlp']['means'], "MLP"
        )
        
        # New plots - Figure 2 style (full comparison)
        full_correlator_gbr_fig = plot_full_correlator_comparison(
            time_values, statistics['truth']['means'], truth_train_means,
            gbr_uncorr_means, statistics['gbr']['means'], "GBR"
        )
        
        full_correlator_mlp_fig = plot_full_correlator_comparison(
            time_values, statistics['truth']['means'], truth_train_means,
            mlp_uncorr_means, statistics['mlp']['means'], "MLP"
        )
        
        full_nts_gbr_fig = plot_full_nts_comparison(
            time_values, statistics['truth']['nts_ratios'], truth_train_nts,
            gbr_uncorr_nts, statistics['gbr']['nts_ratios'], "GBR"
        )
        
        full_nts_mlp_fig = plot_full_nts_comparison(
            time_values, statistics['truth']['nts_ratios'], truth_train_nts,
            mlp_uncorr_nts, statistics['mlp']['nts_ratios'], "MLP"
        )
        
        # New plot - Figure 3 style (fit parameters)
        fit_params_fig = plot_fit_parameter_comparison(fit_results)
        
        # Create output directory for this experiment
        output_dir = f"results_{experiment_label}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all figures in the experiment-specific directory
        print(f"   Saving figures to directory: {output_dir}/")
        correlator_fig.savefig(f"{output_dir}/correlator_comparison.png", dpi=300, bbox_inches='tight')
        nts_fig.savefig(f"{output_dir}/nts_comparison.png", dpi=300, bbox_inches='tight')
        bias_correction_gbr_fig.savefig(f"{output_dir}/bias_correction_effect_gbr.png", dpi=300, bbox_inches='tight')
        bias_correction_mlp_fig.savefig(f"{output_dir}/bias_correction_effect_mlp.png", dpi=300, bbox_inches='tight')
        full_correlator_gbr_fig.savefig(f"{output_dir}/full_correlator_comparison_gbr.png", dpi=300, bbox_inches='tight')
        full_correlator_mlp_fig.savefig(f"{output_dir}/full_correlator_comparison_mlp.png", dpi=300, bbox_inches='tight')
        full_nts_gbr_fig.savefig(f"{output_dir}/full_nts_comparison_gbr.png", dpi=300, bbox_inches='tight')
        full_nts_mlp_fig.savefig(f"{output_dir}/full_nts_comparison_mlp.png", dpi=300, bbox_inches='tight')
        fit_params_fig.savefig(f"{output_dir}/fit_parameter_comparison.png", dpi=300, bbox_inches='tight')
        
        print("   All plots generated and saved:")
        print(f"     - {output_dir}/correlator_comparison.png")
        print(f"     - {output_dir}/nts_comparison.png")
        print(f"     - {output_dir}/bias_correction_effect_gbr.png")
        print(f"     - {output_dir}/bias_correction_effect_mlp.png")
        print(f"     - {output_dir}/full_correlator_comparison_gbr.png")
        print(f"     - {output_dir}/full_correlator_comparison_mlp.png")
        print(f"     - {output_dir}/full_nts_comparison_gbr.png")
        print(f"     - {output_dir}/full_nts_comparison_mlp.png")
        print(f"     - {output_dir}/fit_parameter_comparison.png")
        
        # Display plots
        plt.show()
        
        print("\n" + "=" * 40)
        print(f"Lattice QCD Analysis Complete for experiment: {experiment_label}!")
        print("=" * 40)
        
        # Return results for potential further analysis
        return {
            'statistics': statistics,
            'extended_statistics': {
                'truth_train_means': truth_train_means,
                'truth_train_nts': truth_train_nts,
                'gbr_uncorr_means': gbr_uncorr_means,
                'gbr_uncorr_nts': gbr_uncorr_nts,
                'mlp_uncorr_means': mlp_uncorr_means,
                'mlp_uncorr_nts': mlp_uncorr_nts
            },
            'fit_results': fit_results,
            'models': {'gbr': gbr_model, 'mlp': mlp_model},
            'predictions': {
                'gbr_bc': gbr_pred_bc, 'gbr_ud': gbr_pred_ud,
                'mlp_bc': mlp_pred_bc, 'mlp_ud': mlp_pred_ud
            },
            'data': {
                'input': input_reshaped, 'target': target_reshaped,
                'truth': truth_data, 'time_values': time_values
            }
        }
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("Analysis terminated due to error.")
        raise


if __name__ == "__main__":
    # Only run main if this script is executed directly
    main()