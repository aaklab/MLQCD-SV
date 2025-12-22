# MLQCD-SV: Machine Learning for Lattice QCD

Physics-informed machine learning pipeline for predicting lattice QCD correlators.

## Project Structure

```
MLQCD-SV/
├── src/                          # Source code
│   ├── config.py                 # Configuration settings
│   ├── lattice_qcd_analysis.py   # Main analysis pipeline
│   ├── training.py               # ML model training
│   ├── physics.py                # Physics calculations
│   ├── plotting.py               # Visualization
│   ├── data_prep.py              # Data preprocessing
│   ├── gbr_pinns.py              # GBR+PINNS implementation
│   ├── run_experiment.py         # Experiment runner
│   ├── aggregate_spectral_results.py  # Results aggregation
│   └── utils/                    # Utility scripts
│       ├── clean_results.py      # Results cleaning
│       └── combine_results.py    # Results combination
│
├── bat/                          # Batch scripts for Windows
│   ├── interactive_analysis.bat  # Interactive menu (recommended)
│   ├── run_spectral_analysis.bat # Batch analysis on 7 datasets
│   ├── run_combine.bat           # Combine results
│   └── extract_parameters.bat    # Extract parameters
│
├── data/raw/                     # Input data (CSV files)
├── results/batch/                # Batch analysis results
├── docs/                         # Documentation and papers
│
└── README.md                     # This file
```

## Quick Start

### Interactive Analysis (Recommended)

```cmd
bat\interactive_analysis.bat
```

This opens a menu where you can:
- Select individual datasets
- Choose specific ML models
- Run focused analysis

### Run Analysis on 7 Datasets

```cmd
bat\run_spectral_analysis.bat
```

This will:
- Run GBR model on 7 experiments
- Generate spectral fit parameters
- Create PDF reports with plots
- Save results to `results/batch/`

### Combine Results

```cmd
bat\run_combine.bat
```

Creates a combined CSV file with spectral fit parameters from all experiments.

## Available ML Methods

- **GBR**: Gradient Boosting Regressor (default)
- **MLP**: Multi-Layer Perceptron
- **Ridge**: Ridge Regression
- **DTREE**: Decision Tree
- **CNN**: Convolutional Neural Network (requires PyTorch)
- **Transformer**: Transformer model (requires PyTorch)
- **RM+GBR**: Ratio Method + GBR (Vega's technique)
- **GBR+PINNS**: Physics-Informed Neural Networks with GBR preprocessing

## Configuration

Edit `src/config.py` to:
- Select models: `RUN_MODELS = ["GBR", "MLP", ...]`
- Enable/disable methods: `ENABLE_RATIO_METHOD`, `ENABLE_GBR_PINNS`
- Adjust hyperparameters
- Configure preprocessing options

## Key Features

- **Bias Correction**: Additive bias correction using separate data partitions
- **Bayesian Fitting**: MCMC-based spectral parameter extraction with priors
- **Ratio Method**: Vega's RM+ML technique for improved predictions
- **Physics-Informed**: GBR+PINNS with spectral ansatz constraints
- **Effective Mass Analysis**: Ground state energy extraction
- **Comprehensive Plotting**: Automated PDF generation with all analysis plots

## Output Files

- `spectral_fit_parameters.txt`: Detailed fit parameters for each experiment
- `summary_plots_*.pdf`: Combined PDF with all plots
- `combined_spectral_parameters.csv`: Aggregated results from all experiments

## Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib
- scikit-learn
- SciPy
- PyTorch (optional, for CNN/Transformer/GBR+PINNS)

## Citation

If you use this code, please cite the relevant papers in the `docs/` folder.