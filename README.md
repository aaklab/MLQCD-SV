# Machine Learning Model Comparison Project

This project implements and compares multiple machine learning models for regression tasks, including MLP, Gradient Boosting Regressor, CNN, and Transformer architectures.

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── configs/
│   ├── mlp.yaml
│   ├── gbr.yaml
│   ├── cnn.yaml
│   └── transformer.yaml
├── src/
│   ├── preprocessing/
│   │   └── preprocess_data.py
│   ├── models/
│   │   ├── mlp_model.py
│   │   ├── gbr_model.py
│   │   ├── cnn_model.py
│   │   └── transformer_model.py
│   ├── training/
│   │   ├── train_mlp.py
│   │   ├── train_gbr.py
│   │   ├── train_cnn.py
│   │   └── train_transformer.py
│   ├── evaluation/
│   │   ├── bias_correction.py
│   │   ├── evaluate_models.py
│   │   └── spectral_fit.py
│   └── utils.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── MLP_experiments.ipynb
│   ├── GBR_experiments.ipynb
│   ├── CNN_experiments.ipynb
│   ├── Transformer_experiments.ipynb
│   └── results_comparison.ipynb
├── results/
│   ├── figures/
│   ├── metrics/
│   └── checkpoints/
└── docs/
    ├── paper_summary.md
    └── notes.md
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Your Data**
   - Place your dataset in `data/raw/`
   - Update configuration files in `configs/` with your data specifications

3. **Train Models**
   ```bash
   # Train MLP
   python src/training/train_mlp.py --config configs/mlp.yaml --data data/raw/your_dataset.csv
   
   # Train GBR
   python src/training/train_gbr.py --config configs/gbr.yaml --data data/raw/your_dataset.csv
   
   # Train CNN (for image data)
   python src/training/train_cnn.py --config configs/cnn.yaml --data data/raw/your_dataset.csv
   
   # Train Transformer (for sequence data)
   python src/training/train_transformer.py --config configs/transformer.yaml --data data/raw/your_dataset.csv
   ```

4. **Compare Results**
   - Use the `notebooks/results_comparison.ipynb` notebook for comprehensive analysis
   - Or generate evaluation report programmatically

## Models Implemented

### 1. Multi-Layer Perceptron (MLP)
- Configurable architecture with multiple hidden layers
- Dropout regularization
- Early stopping
- TensorFlow/Keras implementation

### 2. Gradient Boosting Regressor (GBR)
- Hyperparameter optimization with Optuna
- Cross-validation
- Feature importance analysis
- Scikit-learn implementation

### 3. Convolutional Neural Network (CNN)
- Configurable convolutional layers
- Data augmentation support
- Suitable for image-based regression tasks
- TensorFlow/Keras implementation

### 4. Transformer
- Multi-head attention mechanism
- Positional encoding
- Configurable architecture
- Suitable for sequence-based regression tasks

## Features

- **Configuration-driven**: All models use YAML configuration files
- **Comprehensive evaluation**: Multiple metrics, bias correction, statistical testing
- **Reproducible**: Seed management and logging
- **Modular design**: Easy to extend with new models
- **Jupyter notebooks**: Interactive analysis and experimentation
- **Bias correction**: Multiple methods for improving predictions
- **Spectral analysis**: Specialized tools for astronomical data

## Configuration

Each model has its own configuration file in the `configs/` directory. Key parameters include:

- Model architecture (layers, units, etc.)
- Training parameters (batch size, epochs, learning rate)
- Data preprocessing options
- Paths for saving models and logs

## Evaluation

The project includes comprehensive evaluation tools:

- **Multiple metrics**: RMSE, MAE, R², MAPE
- **Bias correction**: Linear, polynomial, and quantile-based methods
- **Statistical testing**: Significance tests between models
- **Visualization**: Prediction plots, residual analysis, feature importance

## Contributing

1. Follow the existing code structure and naming conventions
2. Add appropriate logging and error handling
3. Update configuration files for new models
4. Add corresponding notebook for experimentation
5. Update documentation

## License

[Add your license here]
