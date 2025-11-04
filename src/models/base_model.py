# Template class that all models follow
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all ML models"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
        
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        pass
        
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
        
    @abstractmethod
    def save_model(self, filepath):
        """Save trained model"""
        pass
        
    @abstractmethod
    def load_model(self, filepath):
        """Load trained model"""
        pass