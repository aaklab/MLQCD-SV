#!/usr/bin/env python3
"""
Gradient Boosting Regressor Model Implementation

Implements GBR for predicting physics parameters (E₀, A₀) from correlator data.
Well-suited for tabular time-series data with non-linear relationships.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel


class GBRModel(BaseModel):
    """Gradient Boosting Regressor model for physics parameter prediction"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.training_history = {}
        
    def build_model(self):
        """Build the GBR model with configuration parameters"""
        model_config = self.config.get('model', {})
        
        # Create base GBR
        gbr = GradientBoostingRegressor(
            n_estimators=model_config.get('n_estimators', 100),
            learning_rate=model_config.get('learning_rate', 0.1),
            max_depth=model_config.get('max_depth', 6),
            subsample=model_config.get('subsample', 0.8),
            random_state=self.config.get('fit_options', {}).get('seed', 42),
            verbose=1  # Show training progress
        )
        
        # Wrap in MultiOutputRegressor for multiple targets (E₀, A₀)
        self.model = MultiOutputRegressor(gbr)
        
        print(f"🌳 Built GBR model:")
        print(f"   N estimators: {model_config.get('n_estimators', 100)}")
        print(f"   Learning rate: {model_config.get('learning_rate', 0.1)}")
        print(f"   Max depth: {model_config.get('max_depth', 6)}")
        print(f"   Subsample: {model_config.get('subsample', 0.8)}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the GBR model
        
        Args:
            X_train: Training correlator data (N_samples, N_time_slices)
            y_train: Training physics parameters (N_samples, 2) [E₀, A₀]
            X_val: Validation correlator data
            y_val: Validation physics parameters
            
        Returns:
            Training history dictionary
        """
        print("🚂 TRAINING GBR MODEL")
        print("=" * 30)
        
        if self.model is None:
            self.build_model()
        
        # Handle NaN values in input data
        X_train_clean, y_train_clean = self._clean_data(X_train, y_train)
        X_val_clean, y_val_clean = self._clean_data(X_val, y_val)
        
        print(f"   Training samples: {X_train_clean.shape[0]}")
        print(f"   Validation samples: {X_val_clean.shape[0]}")
        print(f"   Features: {X_train_clean.shape[1]}")
        print(f"   Targets: {y_train_clean.shape[1]} (E₀, A₀)")
        
        # Train the model
        print("\n   Training in progress...")
        self.model.fit(X_train_clean, y_train_clean)
        
        # Evaluate on training and validation sets
        train_pred = self.model.predict(X_train_clean)
        val_pred = self.model.predict(X_val_clean)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train_clean, train_pred)
        train_mae = mean_absolute_error(y_train_clean, train_pred)
        val_mse = mean_squared_error(y_val_clean, val_pred)
        val_mae = mean_absolute_error(y_val_clean, val_pred)
        
        # Store training history
        self.training_history = {
            'train_mse': float(train_mse),
            'train_mae': float(train_mae),
            'val_mse': float(val_mse),
            'val_mae': float(val_mae),
            'train_samples': X_train_clean.shape[0],
            'val_samples': X_val_clean.shape[0]
        }
        
        # Print results
        print(f"\n📊 TRAINING RESULTS:")
        print(f"   Train MSE: {train_mse:.6f}")
        print(f"   Train MAE: {train_mae:.6f}")
        print(f"   Val MSE: {val_mse:.6f}")
        print(f"   Val MAE: {val_mae:.6f}")
        
        # Feature importance analysis
        self._analyze_feature_importance()
        
        return self.training_history
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input correlator data (N_samples, N_time_slices)
            
        Returns:
            Predicted physics parameters (N_samples, 2) [E₀, A₀]
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Clean input data
        X_clean = self._clean_input_data(X)
        
        # Make predictions
        predictions = self.model.predict(X_clean)
        
        return predictions
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'config': self.config,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model and metadata
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.training_history = model_data.get('training_history', {})
        
        print(f"📂 Model loaded from: {filepath}")
    
    def _clean_data(self, X, y):
        """Remove samples with NaN values"""
        # Find samples with any NaN values
        X_nan_mask = np.isnan(X).any(axis=1)
        y_nan_mask = np.isnan(y).any(axis=1)
        valid_mask = ~(X_nan_mask | y_nan_mask)
        
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if valid_mask.sum() < len(valid_mask):
            removed = len(valid_mask) - valid_mask.sum()
            print(f"   Removed {removed} samples with NaN values")
        
        return X_clean, y_clean
    
    def _clean_input_data(self, X):
        """Clean input data for prediction (replace NaN with mean)"""
        X_clean = X.copy()
        
        # Replace NaN with column means
        nan_mask = np.isnan(X_clean)
        if nan_mask.any():
            col_means = np.nanmean(X_clean, axis=0)
            for col in range(X_clean.shape[1]):
                X_clean[nan_mask[:, col], col] = col_means[col]
        
        return X_clean
    
    def _analyze_feature_importance(self):
        """Analyze and print feature importance (which time slices matter most)"""
        if hasattr(self.model, 'estimators_'):
            # Get feature importance from the first estimator (E₀ predictor)
            e0_estimator = self.model.estimators_[0]
            a0_estimator = self.model.estimators_[1]
            
            e0_importance = e0_estimator.feature_importances_
            a0_importance = a0_estimator.feature_importances_
            
            print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
            print(f"   Top 5 time slices for E₀ prediction:")
            e0_top_indices = np.argsort(e0_importance)[-5:][::-1]
            for i, idx in enumerate(e0_top_indices):
                print(f"     {i+1}. t={idx}: {e0_importance[idx]:.4f}")
            
            print(f"   Top 5 time slices for A₀ prediction:")
            a0_top_indices = np.argsort(a0_importance)[-5:][::-1]
            for i, idx in enumerate(a0_top_indices):
                print(f"     {i+1}. t={idx}: {a0_importance[idx]:.4f}")
            
            # Store importance for later analysis
            self.training_history['feature_importance'] = {
                'e0_importance': e0_importance.tolist(),
                'a0_importance': a0_importance.tolist()
            }