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