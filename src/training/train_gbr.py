#!/usr/bin/env python3
"""
Training Script for Gradient Boosting Regressor - Step 3

Trains GBR model to predict physics parameters (E₀, A₀) from correlator data.
This implements the surrogate model approach for lattice QCD analysis.
"""

import numpy as np
import yaml
import json
import os
from datetime import datetime
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gbr_model import GBRModel
from preprocessing.preprocess_data import load_processed_data
from utils import create_experiment_structure


def load_config(config_path="configs/gbr.yaml", base_config_path="configs/base.yaml"):
    """Load configuration from YAML files"""
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load model-specific config
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Merge configs (model config overrides base)
    config = {**base_config, **model_config}
    
    return config


def create_target_labels_from_individual_fits(processed_data_indices, 
                                            individual_results_path="results/metrics/individual_fit_results.json"):
    """
    Create varied target labels (E₀, A₀) from individual configuration fits.
    
    This uses the individual fitting results to create proper varied targets
    for ML training instead of identical targets.
    
    Args:
        processed_data_indices: Indices of configurations after preprocessing
        individual_results_path: Path to individual fitting results
        
    Returns:
        Target array (n_samples, 2) with varied [E₀, A₀] values
    """
    print(f"📊 CREATING VARIED TARGET LABELS FROM INDIVIDUAL FITS")
    
    # Load individual fitting results
    with open(individual_results_path, 'r') as f:
        individual_data = json.load(f)
    
    individual_results = individual_data['individual_fits']
    print(f"   Loaded {len(individual_results)} individual fit results")
    
    # Create mapping from config_idx to fit results
    fit_results_map = {result['config_idx']: result for result in individual_results}
    
    # Create target arrays
    targets = []
    missing_configs = 0
    
    for idx in processed_data_indices:
        if idx in fit_results_map:
            result = fit_results_map[idx]
            targets.append([result['E0'], result['A0']])
        else:
            # Use mean values for missing configurations
            mean_E0 = np.mean([r['E0'] for r in individual_results])
            mean_A0 = np.mean([r['A0'] for r in individual_results])
            targets.append([mean_E0, mean_A0])
            missing_configs += 1
    
    targets = np.array(targets)
    
    print(f"   Created {len(targets)} varied target labels")
    print(f"   Missing configs (using mean): {missing_configs}")
    print(f"   E₀ range: [{targets[:, 0].min():.6f}, {targets[:, 0].max():.6f}]")
    print(f"   A₀ range: [{targets[:, 1].min():.6f}, {targets[:, 1].max():.6f}]")
    print(f"   E₀ std: {targets[:, 0].std():.6f}")
    print(f"   A₀ std: {targets[:, 1].std():.6f}")
    
    return targets


def create_target_labels(n_samples, benchmark_results_path="results/metrics/benchmark_fit_results.json"):
    """
    Create target labels (E₀, A₀) for training.
    
    DEPRECATED: This creates identical targets. Use create_target_labels_from_individual_fits instead.
    """
    print(f"📊 CREATING TARGET LABELS (DEPRECATED - IDENTICAL TARGETS)")
    
    # Load benchmark results from Step 1
    with open(benchmark_results_path, 'r') as f:
        benchmark_results = json.load(f)
    
    fitted_params = benchmark_results['fitted_params']
    E0_target = fitted_params['E0']['value']
    A0_target = fitted_params['A0']['value']
    
    print(f"   Target E₀: {E0_target:.6f}")
    print(f"   Target A₀: {A0_target:.6f}")
    print(f"   Creating {n_samples} identical target labels")
    
    # Create target array - all samples have same target for single-dataset training
    targets = np.full((n_samples, 2), [E0_target, A0_target])
    
    return targets


def train_gbr_model(config_path="configs/gbr.yaml"):
    """
    Main training function for GBR model
    
    Args:
        config_path: Path to GBR configuration file
        
    Returns:
        Trained model and results
    """
    print("🌳 GBR TRAINING PIPELINE - STEP 3")
    print("=" * 50)
    
    # Load configuration
    config = load_config(config_path)
    print(f"📋 Loaded configuration from: {config_path}")
    
    # Load processed data from Step 2
    print(f"\n📁 LOADING PROCESSED DATA")
    data_dir = config['data']['processed_path']
    splits, metadata = load_processed_data(data_dir)
    
    X_train = splits['train']
    X_val = splits['val']
    X_test = splits['test']
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Create varied target labels from individual fits
    # Note: We need to map processed data indices back to original config indices
    # For now, assume first N configs were used (this could be improved)
    
    try:
        # Try to use varied targets from individual fits
        train_indices = list(range(X_train.shape[0]))
        val_indices = list(range(X_train.shape[0], X_train.shape[0] + X_val.shape[0]))
        test_indices = list(range(X_train.shape[0] + X_val.shape[0], 
                                X_train.shape[0] + X_val.shape[0] + X_test.shape[0]))
        
        y_train = create_target_labels_from_individual_fits(train_indices)
        y_val = create_target_labels_from_individual_fits(val_indices)
        y_test = create_target_labels_from_individual_fits(test_indices)
        
        print("✅ Using varied targets from individual fits")
        
    except FileNotFoundError:
        print("⚠️  Individual fit results not found, using identical targets")
        y_train = create_target_labels(X_train.shape[0])
        y_val = create_target_labels(X_val.shape[0])
        y_test = create_target_labels(X_test.shape[0])
    
    # Initialize and train model
    print(f"\n🏗️  INITIALIZING GBR MODEL")
    model = GBRModel(config)
    
    # Train the model
    training_history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print(f"\n🧪 EVALUATING ON TEST SET")
    test_predictions = model.predict(X_test)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    
    print(f"   Test MSE: {test_mse:.6f}")
    print(f"   Test MAE: {test_mae:.6f}")
    
    # Add test metrics to history
    training_history['test_mse'] = float(test_mse)
    training_history['test_mae'] = float(test_mae)
    
    # Save model and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment', {}).get('name', 'gbr_experiment')
    output_base_dir = config.get('experiment', {}).get('output_dir', 'results/runs')
    output_dir = os.path.join(output_base_dir, f"{experiment_name}_{timestamp}")
    
    # Create experiment structure with subfolders
    experiment_paths = create_experiment_structure(output_dir)
    
    # Save trained model to checkpoints folder
    model_path = os.path.join(experiment_paths['checkpoints'], "gbr_model.joblib")
    model.save_model(model_path)
    
    # Save training results to metrics folder
    results = {
        'timestamp': timestamp,
        'config': config,
        'training_history': training_history,
        'data_shapes': {
            'train': list(X_train.shape),
            'val': list(X_val.shape),
            'test': list(X_test.shape)
        },
        'test_predictions_sample': test_predictions[:5].tolist(),  # First 5 predictions
        'target_values': {
            'E0': float(y_test[0, 0]),
            'A0': float(y_test[0, 1])
        }
    }
    
    results_path = os.path.join(experiment_paths['metrics'], "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 RESULTS SAVED")
    print(f"   Model: {model_path}")
    print(f"   Results: {results_path}")
    print(f"   Output directory: {output_dir}")
    print(f"   Subfolders created: metrics/, plots/, checkpoints/")
    
    # Physics validation
    print(f"\n🔬 PHYSICS VALIDATION")
    print(f"   Target E₀: {y_test[0, 0]:.6f}")
    print(f"   Predicted E₀ (mean): {test_predictions[:, 0].mean():.6f} ± {test_predictions[:, 0].std():.6f}")
    print(f"   Target A₀: {y_test[0, 1]:.6f}")
    print(f"   Predicted A₀ (mean): {test_predictions[:, 1].mean():.6f} ± {test_predictions[:, 1].std():.6f}")
    
    print(f"\n✅ GBR TRAINING COMPLETED")
    print(f"   Surrogate model ready for bias correction (Step 4)")
    
    return model, results


if __name__ == "__main__":
    # Run GBR training
    model, results = train_gbr_model()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review training results in: {results['timestamp']}")
    print(f"   2. Proceed to Step 4: Bias correction")
    print(f"   3. Compare with traditional fitting (Step 5)")