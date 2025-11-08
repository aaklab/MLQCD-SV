# Seed_everything(), timestamp helpers, etc.
import random
import numpy as np
import os
from datetime import datetime

def seed_everything(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # If using torch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # If using tensorflow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def get_timestamp():
    """Get current timestamp in YYYY-MM-DD format"""
    return datetime.now().strftime("%Y-%m-%d")

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def create_experiment_structure(experiment_dir):
    """
    Create standardized experiment folder structure.
    
    Creates the main experiment directory and required subfolders:
    - metrics/: for storing evaluation metrics and results
    - plots/: for storing visualization outputs  
    - checkpoints/: for storing model checkpoints
    
    Args:
        experiment_dir (str): Path to the main experiment directory
        
    Returns:
        dict: Dictionary with paths to created subdirectories
    """
    # Create main experiment directory
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        'metrics': os.path.join(experiment_dir, 'metrics'),
        'plots': os.path.join(experiment_dir, 'plots'), 
        'checkpoints': os.path.join(experiment_dir, 'checkpoints')
    }
    
    for subdir_name, subdir_path in subdirs.items():
        os.makedirs(subdir_path, exist_ok=True)
    
    # Add the main directory to the returned paths
    subdirs['main'] = experiment_dir
    
    return subdirs