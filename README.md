# ML for Lattice QCD

This project implements and compares multiple machine learning models for lattice QCD correlator analysis, including MLP, Gradient Boosting Regressor, CNN, and Transformer architectures.

## Project Structure

```
ML_for_lattice_QCD/
├── README.md
├── requirements.txt
├── analysis/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   └── results_comparison.ipynb
├── configs/
│   ├── base.yaml
│   ├── cnn.yaml
│   ├── gbr.yaml
│   ├── mlp.yaml
│   └── transformer.yaml
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── notes.md
│   └── paper_summary.md
├── results/
│   ├── checkpoints/
│   ├── figures/
│   ├── metrics/
│   └── runs/
└── src/
    ├── data/
    ├── evaluation/
    ├── models/
    ├── pipeline/
    ├── preprocessing/
    ├── training/
    └── utils.py
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Your Data**
   - Place your .gpl files in `data/raw/`
   - Update configuration files in `configs/` with your data specifications

3. **Run Experiments**
   ```bash
   # Run MLP experiment
   python src/pipeline/run_experiment.py configs/mlp.yaml
   
   # Run GBR experiment
   python src/pipeline/run_experiment.py configs/gbr.yaml
   
   # Run CNN experiment
   python src/pipeline/run_experiment.py configs/cnn.yaml
   
   # Run Transformer experiment
   python src/pipeline/run_experiment.py configs/transformer.yaml
   ```

4. **Analyze Results**
   - Each experiment creates a timestamped folder in `results/runs/`
   - Use `analysis/results_comparison.ipynb` for comprehensive comparison
   - Explore data with `analysis/data_exploration.ipynb`

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

- **Automated Experiment Management**: Each run creates its own timestamped folder with all results
- **Configuration Hierarchy**: Base config with model-specific overrides
- **Lattice QCD Focused**: Specialized tools for .gpl file loading and correlator analysis
- **Comprehensive Evaluation**: Multiple metrics, bias correction, spectral fitting
- **Reproducible**: Seed management and logging
- **Modular Design**: Factory pattern for easy model extension
- **Interactive Analysis**: Jupyter notebooks for exploration and comparison

## Configuration

The configuration system uses a hierarchical approach:

- **base.yaml**: Shared settings like data paths, fit options, and experiment settings
- **Model configs**: Override specific parameters for each model type

Key configuration sections:
- `data`: Paths to raw and processed data
- `fit_options`: Training parameters like validation split, seed
- `model`: Architecture and hyperparameters
- `experiment`: Name and output directory

Example experiment folder structure:
```
results/runs/2025-11-04__mlp/
├── config.yaml          # Complete config used for this run
├── metrics.json         # Evaluation metrics
├── plots/              # Generated visualizations
├── artifacts/          # Saved models and other outputs
└── logs.txt           # Training logs
```

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
