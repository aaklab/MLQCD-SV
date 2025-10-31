# Project Notes

## Development Log

### [Date] - Project Setup
- Created project structure
- Implemented base model classes
- Set up configuration files
- Created preprocessing pipeline

### [Date] - Model Implementation
- Implemented MLP model with TensorFlow/Keras
- Implemented GBR model with scikit-learn
- Implemented CNN model for image-based tasks
- Implemented Transformer model for sequence data

### [Date] - Evaluation Framework
- Created comprehensive evaluation metrics
- Implemented bias correction methods
- Added spectral fitting utilities
- Created model comparison framework

## Technical Notes

### Data Preprocessing
- **Normalization**: Standard scaling vs MinMax scaling
- **Train/Val/Test Split**: 60/20/20 split with stratification
- **Missing Values**: Handled by dropping (consider imputation for future work)

### Model Architecture Decisions

#### MLP
- Hidden layers: [128, 64, 32] - good balance of capacity and overfitting prevention
- Dropout: 0.2 - helps with regularization
- Activation: ReLU - standard choice for hidden layers

#### GBR
- n_estimators: 100 - good starting point, can be tuned
- learning_rate: 0.1 - conservative to prevent overfitting
- max_depth: 6 - moderate depth to capture interactions

#### CNN
- Conv layers: Progressive filter increase (32→64→128)
- Pooling: MaxPooling2D after each conv layer
- Dense layers: [256, 128] before output

#### Transformer
- d_model: 128 - sufficient for most tasks
- num_heads: 8 - standard choice
- num_layers: 6 - deep enough for complex patterns

### Hyperparameter Tuning
- Used Optuna for GBR optimization
- Early stopping for neural networks
- Learning rate scheduling for Transformer

### Performance Insights
- [Add insights as you discover them]
- [Model comparison results]
- [Best practices learned]

## Issues and Solutions

### Issue 1: [Description]
**Solution**: [How it was resolved]

### Issue 2: [Description]
**Solution**: [How it was resolved]

## TODO Items
- [ ] Implement ensemble methods
- [ ] Add cross-validation to neural networks
- [ ] Implement feature importance for neural networks
- [ ] Add more sophisticated data augmentation
- [ ] Implement attention visualization for Transformer
- [ ] Add model interpretability tools

## Useful Commands

### Training Models
```bash
# Train MLP
python src/training/train_mlp.py --config configs/mlp.yaml --data data/processed/dataset.csv

# Train GBR
python src/training/train_gbr.py --config configs/gbr.yaml --data data/processed/dataset.csv

# Train CNN
python src/training/train_cnn.py --config configs/cnn.yaml --data data/processed/dataset.csv

# Train Transformer
python src/training/train_transformer.py --config configs/transformer.yaml --data data/processed/dataset.csv
```

### Evaluation
```bash
# Generate evaluation report
python -c "
from src.evaluation.evaluate_models import ModelEvaluator
evaluator = ModelEvaluator()
evaluator.generate_evaluation_report([
    'results/mlp_results.json',
    'results/gbr_results.json',
    'results/cnn_results.json',
    'results/transformer_results.json'
])
"
```

## Research Ideas
- Investigate domain-specific architectures
- Explore transfer learning approaches
- Consider multi-task learning if multiple targets available
- Investigate uncertainty quantification methods

## Code Quality Notes
- All models follow consistent interface
- Configuration-driven approach for reproducibility
- Comprehensive logging and error handling
- Modular design for easy extension