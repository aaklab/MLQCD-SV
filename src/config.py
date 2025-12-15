"""
Clean and modern configuration file for the MLQCD-SV analysis pipeline.
Only simple constants and preprocessing flags live here.
No legacy flags, no duplicated settings.
"""

import os

# ------------------------------------------------------------------
# 1. Preprocessing configuration (NEW unified system)
# ------------------------------------------------------------------
# Preprocessing / variance reduction
CONTROL_VARIATE_ENABLED = False

# Apply symmetry C(t) = C(T - t)
PRE_SYMMETRY = True

# Apply time window [MIN, MAX) to both input and target correlators
PRE_TIME_WINDOW = False
PRE_TIME_WINDOW_MIN = 0
PRE_TIME_WINDOW_MAX = 60     # -> uses t = 0..59

# Normalisation:
#   "none"   – no normalisation (recommended for GBR)
#   "l2"     – row-wise L2 normalisation
#   "zscore" – per-time-slice z-score normalisation
PRE_NORMALISATION = "none"

# If True: only normalise input (X), leave target (Y) untouched
PRE_NORMALISE_INPUT_ONLY = False

# Log-scaling of correlators (rarely needed)
PRE_LOG_ABS = False
PRE_LOG_EPS = 1e-12

# ------------------------------------------------------------------
# 2. Debugging
# ------------------------------------------------------------------

# When True: use only the first N configs for quick runs
DEBUG_FAST = True
DEBUG_FAST_N_CONFIGS = 30

# Add timestamp to output PDF
TIMESTAMP_PDFS = True

# Global random seed
RANDOM_SEED = 42

# ------------------------------------------------------------------
# 3. Model selection
# ------------------------------------------------------------------

RUN_MODELS = ["GBR", "MLP", "RIDGE", "DTREE", "CNN", "TRANSFORMER"]

# Wrap regressors in MultiOutputRegressor
USE_MULTI_OUTPUT = True

# Ridge Regression
RIDGE_ALPHA = 1.0

# Decision Tree Regression
DTREE_MAX_DEPTH = 5
DTREE_MIN_SAMPLES_LEAF = 5

# ------------------------------------------------------------------
# 4. Time-source partitions (fixed by LQCD data generation)
# ------------------------------------------------------------------

N_TIME_SOURCES = 4

TRAIN_SOURCES = [0, 1]
BC_SOURCE = [2]
UD_SOURCE = [3]

# ------------------------------------------------------------------
# 5. Bias-correction and plotting constants
# ------------------------------------------------------------------

TAU_MIN = 5
TAU_MAX = 60
TRUTH_MAGNITUDE_THRESHOLD = 1e-7
BIAS_PLOT_Y_LIMITS = (-10, 10)

# ------------------------------------------------------------------
# 6. Experiment definitions
# ------------------------------------------------------------------

DATA_DIR = "../data/raw"

EXPERIMENTS: dict[int, dict] = {}
_next_experiment_id = 1

def _add_experiment(label: str, exp_type: str, input_file: str, target_file: str) -> None:
    global _next_experiment_id
    EXPERIMENTS[_next_experiment_id] = {
        "label": label,
        "type": exp_type,
        "input_file": os.path.join(DATA_DIR, input_file),
        "target_file": os.path.join(DATA_DIR, target_file),
    }
    _next_experiment_id += 1

# ------------------
# 2-point experiments
# ------------------

_add_experiment("K_ll_to_qsq0",        "2pt", "2pt_K_fine_ll.csv",          "2pt_K_fine_qsq0_ll.csv")
_add_experiment("K_ll_to_qsqmaxby3",   "2pt", "2pt_K_fine_ll.csv",          "2pt_K_fine_qsqmaxby3_ll.csv")
_add_experiment("K_ll_to_2qsqmaxby3",  "2pt", "2pt_K_fine_ll.csv",          "2pt_K_fine_2qsqmaxby3_ll.csv")
_add_experiment("D_Gold_to_nongold",   "2pt", "2pt_D_Gold_fine_ll.csv",     "2pt_D_nongold_fine_ll.csv")

# ------------------
# 3-point experiments
# ------------------

T_VALUES = ["T16", "T19", "T22", "T25"]
Q_OPTIONS = {
    "qsq0": "qsq0",
    "qsqmaxby3": "qsqmaxby3",
    "2qsqmaxby3": "2qsqmaxby3",
}

# localscalar
for T in T_VALUES:
    input_name = f"localscalar_3pt_{T}_fine_ll.csv"
    for q_label, q_suffix in Q_OPTIONS.items():
        _add_experiment(
            label=f"localscalar_{T}_to_{q_label}",
            exp_type="3pt",
            input_file=input_name,
            target_file=f"localscalar_3pt_{T}_fine_{q_suffix}_ll.csv",
        )

# localtempvector
for T in T_VALUES:
    input_name = f"localtempvector_3pt_{T}_fine_ll.csv"
    for q_label, q_suffix in Q_OPTIONS.items():
        _add_experiment(
            label=f"localtempvector_{T}_to_{q_label}",
            exp_type="3pt",
            input_file=input_name,
            target_file=f"localtempvector_3pt_{T}_fine_{q_suffix}_ll.csv",
        )

# ------------------------------------------------------------------
# 7. Gradient Boosting Regressor hyperparameters
# ------------------------------------------------------------------

GBR_MODE              = "balanced"
GBR_N_ESTIMATORS      = 800      # many shallow trees
GBR_LEARNING_RATE     = 0.02     # small step size
GBR_MAX_DEPTH         = 3
GBR_MIN_SAMPLES_SPLIT = 10
GBR_MIN_SAMPLES_LEAF  = 5
GBR_SUBSAMPLE         = 0.8
GBR_N_ITER_NO_CHANGE  = 10
GBR_VALIDATION_FRACTION = 0.1
GBR_N_JOBS            = -1
GBR_LOSS              = "squared_error"


# ---------------------------------------------------------
# 7b. Multi-Layer Perceptron hyperparameters
# ---------------------------------------------------------

MLP_HIDDEN_LAYERS        = (256, 256, 128)     # or (128, 64)
MLP_ACTIVATION           = "relu"
MLP_SOLVER               = "adam"
MLP_ALPHA                = 1e-4           # L2 regularisation
MLP_LEARNING_RATE        = "adaptive"
MLP_LR_INIT              = 1e-3
MLP_MAX_ITER             = 1500
MLP_TOL                  = 1e-4
MLP_RANDOM_STATE         = RANDOM_SEED
MLP_EARLY_STOPPING       = True
MLP_VALIDATION_FRACTION  = 0.1
MLP_N_ITER_NO_CHANGE     = 20
MLP_SHUFFLE              = True
MLP_BATCH_SIZE           = 64

# Adam optimiser parameters
MLP_BETA_1               = 0.9
MLP_BETA_2               = 0.999
MLP_EPSILON              = 1e-8

MLP_VERBOSE              = True


# ------------------------------------------------------------------
# 8. Dataset splits
# ------------------------------------------------------------------

TRAIN_DATASETS = ["T16", "T19"]
VAL_DATASETS   = ["T19"]
TEST_DATASETS  = ["T22", "T25"]

# ------------------------------------------------------------------
# 9. Ensemble configuration
# ------------------------------------------------------------------

N_ENSEMBLE_MODELS = 1