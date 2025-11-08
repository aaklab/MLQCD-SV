# Main script: loads config, trains model, saves results
import os
import sys
import yaml
import json
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.factory import create_model
from preprocessing.preprocess_data import load_processed_data
from evaluation.metrics import compute_all_metrics
from evaluation.plotting import save_all_plots
from utils import seed_everything, create_experiment_structure

def load_config(config_path, base_config_path="configs/base.yaml"):
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
    
    Args:
        processed_data_indices: Indices of configurations after preprocessing
        individual_results_path: Path to individual fitting results
        
    Returns:
        Target array (n_samples, 2) with varied [E₀, A₀] values
    """
    print(f"📊 Creating varied target labels from individual fits")
    
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
    
    return targets

def run_experiment(config_path):
    """
    Main unified experiment runner for all model types.
    
    This function handles the complete ML pipeline:
    1. Load configuration and set seeds
    2. Create experiment folder structure
    3. Load and prepare data
    4. Create model via factory pattern
    5. Train model
    6. Evaluate and save results
    
    Args:
        config_path: Path to model-specific config file
        
    Returns:
        experiment_dir: Path to created experiment directory
    """
    print("🚀 UNIFIED EXPERIMENT RUNNER")
    print("=" * 50)
    
    # Load configuration
    config = load_config(config_path)
    model_type = config.get('model', {}).get('type', 'unknown')
    print(f"📋 Loaded configuration: {config_path}")
    print(f"🤖 Model type: {model_type}")
    
    # Set seed for reproducibility
    seed = config.get('experiment', {}).get('seed', 42)
    seed_everything(seed)
    print(f"🌱 Set random seed: {seed}")
    
    # Create experiment folder structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment', {}).get('name', f'{model_type}_experiment')
    output_base_dir = config.get('experiment', {}).get('output_dir', 'results/runs')
    experiment_dir = os.path.join(output_base_dir, f"{experiment_name}_{timestamp}")
    
    # Create standardized experiment structure
    experiment_paths = create_experiment_structure(experiment_dir)
    print(f"📁 Created experiment structure: {experiment_dir}")
    print(f"   Subfolders: metrics/, plots/, checkpoints/")
    
    # Save config to experiment folder
    config_save_path = os.path.join(experiment_paths['main'], "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, indent=2)
    
    # Load processed data
    print(f"\n📊 LOADING PROCESSED DATA")
    data_dir = config['data']['processed_path']
    splits, metadata = load_processed_data(data_dir)
    
    X_train = splits['train']
    X_val = splits['val'] 
    X_test = splits['test']
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Generate individual fit results within this experiment
    print(f"\n🔬 GENERATING INDIVIDUAL FIT RESULTS FOR THIS EXPERIMENT")
    
    # Import the individual fitting functionality
    from evaluation.individual_fit import run_individual_fitting
    from evaluation.spectral_fit import run_benchmark_fit
    
    # Generate benchmark fit results for this experiment
    print("   Creating benchmark fit results...")
    
    # Import and run benchmark fitting directly to avoid global file creation
    import pandas as pd
    from evaluation.spectral_fit import fit_correlator
    
    # Load data and run benchmark fit
    df = pd.read_csv("data/raw/2pt_D_Gold_fine_ll.csv", header=None)
    benchmark_results = fit_correlator(df, t_min=5, t_max=30)
    benchmark_path = os.path.join(experiment_paths['metrics'], "benchmark_fit_results.json")
    
    # Save benchmark results to this experiment folder
    import json
    with open(benchmark_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'fitted_params': {
                'E0': {'value': float(benchmark_results['fitted_params']['E0'].mean), 
                       'error': float(benchmark_results['fitted_params']['E0'].sdev)},
                'A0': {'value': float(benchmark_results['fitted_params']['A0'].mean), 
                       'error': float(benchmark_results['fitted_params']['A0'].sdev)}
            } if hasattr(benchmark_results['fitted_params']['E0'], 'mean') else benchmark_results['fitted_params'],
            'fit_range': benchmark_results['fit_range'],
            'chi2_dof': benchmark_results['chi2_dof']
        }, f, indent=2)
    
    # Generate individual fit results for this experiment
    print("   Creating individual fit results...")
    
    # Import and run individual fitting directly to control output location
    from evaluation.spectral_fit import fit_individual_correlators
    
    # Run individual fitting without saving to global location
    individual_results = fit_individual_correlators(df, t_min=5, t_max=30, max_configs=500)
    individual_path = os.path.join(experiment_paths['metrics'], "individual_fit_results.json")
    
    # Save individual results to this experiment folder only
    individual_data = {
        'timestamp': timestamp,
        'n_configurations': len(individual_results),
        'individual_fits': individual_results
    }
    with open(individual_path, 'w') as f:
        json.dump(individual_data, f, indent=2)
    
    print(f"   Individual fit results saved to: {individual_path}")
    
    # Create target labels from the individual fits we just generated
    train_indices = list(range(X_train.shape[0]))
    val_indices = list(range(X_train.shape[0], X_train.shape[0] + X_val.shape[0]))
    test_indices = list(range(X_train.shape[0] + X_val.shape[0], 
                            X_train.shape[0] + X_val.shape[0] + X_test.shape[0]))
    
    y_train = create_target_labels_from_individual_fits(train_indices, individual_path)
    y_val = create_target_labels_from_individual_fits(val_indices, individual_path)
    y_test = create_target_labels_from_individual_fits(test_indices, individual_path)
    
    print("✅ Generated and using varied targets from individual fits")
    
    # Create model via factory
    print(f"\n🏗️  CREATING {model_type.upper()} MODEL")
    model = create_model(model_type, config)
    
    # Train model
    print(f"\n🚂 TRAINING MODEL")
    training_history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print(f"\n🧪 EVALUATING ON TEST SET")
    test_predictions = model.predict(X_test)
    
    # Compute metrics
    test_metrics = compute_all_metrics(y_test, test_predictions)
    print(f"   Test MSE: {test_metrics['mse']:.6f}")
    print(f"   Test MAE: {test_metrics['mae']:.6f}")
    print(f"   Test RMSE: {test_metrics['rmse']:.6f}")
    
    # Save model to checkpoints folder
    model_filename = f"{model_type}_model"
    if model_type == 'gbr':
        model_filename += ".joblib"
    else:
        model_filename += ".pth"
    
    model_path = os.path.join(experiment_paths['checkpoints'], model_filename)
    model.save_model(model_path)
    
    # Save training results to metrics folder
    results = {
        'timestamp': timestamp,
        'model_type': model_type,
        'config': config,
        'training_history': training_history,
        'test_metrics': test_metrics,
        'data_shapes': {
            'train': list(X_train.shape),
            'val': list(X_val.shape),
            'test': list(X_test.shape)
        },
        'test_predictions_sample': test_predictions[:5].tolist(),
        'target_statistics': {
            'E0_mean': float(y_test[:, 0].mean()),
            'E0_std': float(y_test[:, 0].std()),
            'A0_mean': float(y_test[:, 1].mean()),
            'A0_std': float(y_test[:, 1].std())
        }
    }
    
    results_path = os.path.join(experiment_paths['metrics'], "experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save plots (if plotting module is implemented)
    try:
        plot_data = {
            "predictions": test_predictions,
            "true": y_test,
            "history": training_history
        }
        save_all_plots(plot_data, experiment_paths['plots'])
        print(f"📈 Plots saved to: {experiment_paths['plots']}")
    except Exception as e:
        print(f"⚠️  Plotting not available: {e}")
    
    print(f"\n💾 EXPERIMENT COMPLETED")
    print(f"   Model: {model_path}")
    print(f"   Results: {results_path}")
    print(f"   Experiment: {experiment_dir}")
    
    # Physics validation
    print(f"\n🔬 PHYSICS VALIDATION")
    print(f"   Target E₀: {y_test[0, 0]:.6f}")
    print(f"   Predicted E₀ (mean): {test_predictions[:, 0].mean():.6f} ± {test_predictions[:, 0].std():.6f}")
    print(f"   Target A₀: {y_test[0, 1]:.6f}")
    print(f"   Predicted A₀ (mean): {test_predictions[:, 1].mean():.6f} ± {test_predictions[:, 1].std():.6f}")
    
    return experiment_dir

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/pipeline/run_experiment.py <config_path>")
        print("\nExamples:")
        print("  python src/pipeline/run_experiment.py configs/gbr.yaml")
        print("  python src/pipeline/run_experiment.py configs/mlp.yaml")
        print("  python src/pipeline/run_experiment.py configs/cnn.yaml")
        print("  python src/pipeline/run_experiment.py configs/transformer.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        experiment_dir = run_experiment(config_path)
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Review results in: {experiment_dir}")
        print(f"   2. Compare with other models")
        print(f"   3. Proceed to bias correction if needed")
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)