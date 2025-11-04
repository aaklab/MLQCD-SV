# Picks the right model based on config
from .mlp_model import MLPModel
from .gbr_model import GBRModel
from .cnn_model import CNNModel
from .transformer_model import TransformerModel

def create_model(model_type, config):
    """Factory function to create models based on type"""
    models = {
        'mlp': MLPModel,
        'gbr': GBRModel,
        'cnn': CNNModel,
        'transformer': TransformerModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type](config)