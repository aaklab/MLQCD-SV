# Main script: loads config, trains model, saves results
import os
import yaml
import json
from datetime import datetime
from ..models.factory import create_model
from ..data.load_data import load_gpl_files
from ..data.preprocess_data import normalize_data, split_data
from ..evaluation.metrics import compute_all_metrics
from ..evaluation.plotting import save_all_plots
from ..utils import seed_everything, get_timestamp

def create_experiment_folder(base_dir, experiment_name):
    """Create a new experiment folder with timestamp"""
    timestamp = get_timestamp()
    experiment_dir = os.path.join(base_dir, f"{timestamp}__{experiment_name}")
    
    # Create directory structure
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "artifacts"), exist_ok=True)
    
    return experiment_dir

def run_experiment(config_path):
    """Main experiment runner"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    seed_everything(config.get('fit_options', {}).get('seed', 42))
    
    # Create experiment folder
    experiment_dir = create_experiment_folder(
        config['experiment']['output_dir'],
        config['experiment']['name']
    )
    
    # Save config to experiment folder
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load and preprocess data
    # data = load_gpl_files(config['data']['raw_path'])
    # processed_data = normalize_data(data)
    # X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data, **config['fit_options'])
    
    # Create and train model
    # model = create_model(config['model']['type'], config)
    # model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    # predictions = model.predict(X_test)
    # metrics = compute_all_metrics(y_test, predictions)
    
    # Save results
    # metrics_path = os.path.join(experiment_dir, "metrics.json")
    # with open(metrics_path, 'w') as f:
    #     json.dump(metrics, f, indent=2)
    
    # Save plots
    # save_all_plots({"predictions": predictions, "true": y_test}, 
    #                os.path.join(experiment_dir, "plots"))
    
    # Save model
    # model.save_model(os.path.join(experiment_dir, "artifacts", "model"))
    
    print(f"Experiment completed. Results saved to: {experiment_dir}")
    return experiment_dir

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    run_experiment(config_path)