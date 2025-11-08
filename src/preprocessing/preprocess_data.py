#!/usr/bin/env python3
"""
Data Preprocessing Pipeline - Step 2

Transforms raw correlator data into clean tensors ready for ML training:
- Truncates at t > 30 or first negative value
- Applies log-scale transformation
- Normalizes data
- Splits into train/validation/test sets
- Saves processed arrays for ML models

This prepares clean input for ML models following physics standards.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import json
from datetime import datetime


def load_correlator_data(filepath: str) -> pd.DataFrame:
    """
    Load correlator data from CSV file.
    
    Args:
        filepath: Path to correlator CSV file
        
    Returns:
        DataFrame with correlator configurations
    """
    print(f"📁 Loading correlator data from: {filepath}")
    df = pd.read_csv(filepath, header=None)
    print(f"   Loaded {df.shape[0]} configurations × {df.shape[1]} time slices")
    return df


def truncate_correlators(df: pd.DataFrame, t_max: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncate correlators at t_max or first negative value per configuration.
    
    Args:
        df: DataFrame with correlator data
        t_max: Maximum time to keep (default: 30)
        
    Returns:
        Tuple of (truncated_data, truncation_indices)
    """
    print(f"✂️  TRUNCATING CORRELATORS")
    print(f"   Target t_max: {t_max}")
    
    truncated_data = []
    truncation_indices = []
    
    for idx, row in df.iterrows():
        correlator = row.values
        
        # Find first negative or zero value
        negative_mask = correlator <= 0
        if negative_mask.any():
            first_negative = np.where(negative_mask)[0][0]
            actual_t_max = min(t_max, first_negative - 1)
        else:
            actual_t_max = t_max
        
        # Ensure we don't go beyond array bounds
        actual_t_max = min(actual_t_max, len(correlator) - 1)
        
        # Truncate correlator
        truncated_correlator = correlator[:actual_t_max + 1]
        truncated_data.append(truncated_correlator)
        truncation_indices.append(actual_t_max)
    
    # Convert to numpy array with padding if needed
    max_length = max(len(corr) for corr in truncated_data)
    padded_data = np.full((len(truncated_data), max_length), np.nan)
    
    for i, corr in enumerate(truncated_data):
        padded_data[i, :len(corr)] = corr
    
    print(f"   Truncated to max length: {max_length}")
    print(f"   Average truncation point: {np.mean(truncation_indices):.1f}")
    print(f"   Min/Max truncation: {min(truncation_indices)}/{max(truncation_indices)}")
    
    return padded_data, np.array(truncation_indices)


def apply_log_transform(data: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """
    Apply log transformation to correlator data.
    
    Args:
        data: Correlator data array
        epsilon: Small value to avoid log(0)
        
    Returns:
        Log-transformed data
    """
    print(f"📊 APPLYING LOG TRANSFORMATION")
    print(f"   Epsilon (minimum value): {epsilon}")
    
    # Clip values to avoid log(0) or log(negative)
    clipped_data = np.clip(data, epsilon, np.inf)
    
    # Apply log transform where data is not NaN
    log_data = np.full_like(data, np.nan)
    valid_mask = ~np.isnan(data)
    log_data[valid_mask] = np.log(clipped_data[valid_mask])
    
    print(f"   Transformed {valid_mask.sum()} valid data points")
    print(f"   Log range: {np.nanmin(log_data):.3f} to {np.nanmax(log_data):.3f}")
    
    return log_data


def normalize_data(data: np.ndarray, method: str = 'subtract_t0') -> Tuple[np.ndarray, Dict]:
    """
    Normalize correlator data.
    
    Args:
        data: Log-transformed correlator data
        method: Normalization method ('subtract_t0', 'divide_t0', 'standard')
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    print(f"🎯 NORMALIZING DATA")
    print(f"   Method: {method}")
    
    normalized_data = data.copy()
    norm_params = {'method': method}
    
    if method == 'subtract_t0':
        # Subtract log(C(t=0)) from each configuration
        t0_values = data[:, 0]  # First time slice
        valid_t0_mask = ~np.isnan(t0_values)
        
        for i in range(len(data)):
            if valid_t0_mask[i]:
                valid_mask = ~np.isnan(data[i])
                normalized_data[i, valid_mask] = data[i, valid_mask] - t0_values[i]
        
        norm_params['t0_mean'] = np.nanmean(t0_values)
        norm_params['t0_std'] = np.nanstd(t0_values)
        
    elif method == 'divide_t0':
        # This would be done before log transform in practice
        print("   Warning: divide_t0 should be applied before log transform")
        
    elif method == 'standard':
        # Standard normalization (mean=0, std=1)
        valid_mask = ~np.isnan(data)
        if valid_mask.any():
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            normalized_data[valid_mask] = (data[valid_mask] - mean_val) / std_val
            norm_params['mean'] = mean_val
            norm_params['std'] = std_val
    
    print(f"   Normalized range: {np.nanmin(normalized_data):.3f} to {np.nanmax(normalized_data):.3f}")
    
    return normalized_data, norm_params


def create_train_test_splits(data: np.ndarray, test_size: float = 0.2, 
                           val_size: float = 0.1, random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Split data into train/validation/test sets.
    
    Args:
        data: Preprocessed correlator data
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining after test)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with train/val/test splits
    """
    print(f"🔀 CREATING TRAIN/VALIDATION/TEST SPLITS")
    print(f"   Test size: {test_size:.1%}")
    print(f"   Validation size: {val_size:.1%}")
    print(f"   Random state: {random_state}")
    
    # First split: separate test set
    X_temp, X_test = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation from remaining training data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val = train_test_split(
        X_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    splits = {
        'train': X_train,
        'val': X_val,
        'test': X_test
    }
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    return splits


def save_processed_data(splits: Dict[str, np.ndarray], metadata: Dict[str, Any], 
                       output_dir: str = "data/processed") -> None:
    """
    Save processed data arrays and metadata.
    
    Args:
        splits: Dictionary with train/val/test data
        metadata: Processing metadata
        output_dir: Output directory for processed data
    """
    print(f"💾 SAVING PROCESSED DATA")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data arrays
    for split_name, data in splits.items():
        filepath = os.path.join(output_dir, f"{split_name}_data.npy")
        np.save(filepath, data)
        print(f"   Saved {split_name} data: {filepath} ({data.shape})")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "preprocessing_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"   Saved metadata: {metadata_path}")


def run_preprocessing_pipeline(filepath: str = None, t_max: int = 30, 
                             log_epsilon: float = 1e-15, 
                             normalization: str = 'subtract_t0',
                             test_size: float = 0.2, val_size: float = 0.1,
                             random_state: int = 42) -> Dict[str, Any]:
    """
    Run the complete data preprocessing pipeline for Step 2.
    
    Args:
        filepath: Path to correlator CSV file
        t_max: Maximum time to keep
        log_epsilon: Epsilon for log transformation
        normalization: Normalization method
        test_size: Test set fraction
        val_size: Validation set fraction
        random_state: Random seed
        
    Returns:
        Dictionary with processing results and metadata
    """
    print("🔄 DATA PREPROCESSING PIPELINE - STEP 2")
    print("=" * 50)
    
    # Auto-detect file path if not provided
    if filepath is None:
        possible_paths = [
            "data/raw/2pt_D_Gold_fine_ll.csv",
            "../../data/raw/2pt_D_Gold_fine_ll.csv",
            "BSc Project/MLQCD-SV/data/raw/2pt_D_Gold_fine_ll.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        
        if filepath is None:
            raise FileNotFoundError("Could not find correlator data file!")
    
    # Step 1: Load data
    df = load_correlator_data(filepath)
    
    # Step 2: Truncate correlators
    truncated_data, truncation_indices = truncate_correlators(df, t_max)
    
    # Step 3: Apply log transformation
    log_data = apply_log_transform(truncated_data, log_epsilon)
    
    # Step 4: Normalize data
    normalized_data, norm_params = normalize_data(log_data, normalization)
    
    # Step 5: Create train/test splits
    splits = create_train_test_splits(normalized_data, test_size, val_size, random_state)
    
    # Step 6: Prepare metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'source_file': filepath,
        'original_shape': df.shape,
        'preprocessing_params': {
            't_max': t_max,
            'log_epsilon': log_epsilon,
            'normalization': normalization,
            'test_size': test_size,
            'val_size': val_size,
            'random_state': random_state
        },
        'truncation_stats': {
            'mean_truncation': float(np.mean(truncation_indices)),
            'min_truncation': int(np.min(truncation_indices)),
            'max_truncation': int(np.max(truncation_indices))
        },
        'normalization_params': norm_params,
        'final_shapes': {split: data.shape for split, data in splits.items()}
    }
    
    # Step 7: Save processed data
    save_processed_data(splits, metadata)
    
    print("\n✅ PREPROCESSING PIPELINE COMPLETED")
    print("   Clean tensors ready for ML training")
    print(f"   Processed data saved to: data/processed/")
    
    return {
        'splits': splits,
        'metadata': metadata,
        'truncation_indices': truncation_indices
    }


def load_processed_data(data_dir: str = "data/processed") -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load previously processed data.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (data_splits, metadata)
    """
    splits = {}
    
    # Load data arrays
    for split_name in ['train', 'val', 'test']:
        filepath = os.path.join(data_dir, f"{split_name}_data.npy")
        if os.path.exists(filepath):
            splits[split_name] = np.load(filepath)
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "preprocessing_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return splits, metadata


if __name__ == "__main__":
    # Run Step 2: Data preprocessing pipeline
    results = run_preprocessing_pipeline(
        t_max=30,
        normalization='subtract_t0',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )