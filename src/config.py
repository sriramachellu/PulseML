"""Paths, dataset constants and hyperparameters."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PARQUET_DIR = OUTPUTS_DIR / "parquet"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

DS1_PARQUET = PARQUET_DIR / "beats_ds1.parquet"
DS2_PARQUET = PARQUET_DIR / "beats_ds2.parquet"
STATS_PARQUET = PARQUET_DIR / "pipeline_stats.parquet"
PREDICTIONS_PARQUET = PARQUET_DIR / "ds2_predictions.parquet"
DEMO_PARQUET = PARQUET_DIR / "demo_beats.parquet"  # small DS2 sample, committed for the dashboard

CNN_CHECKPOINT = MODELS_DIR / "cnn_best.pt"
HYBRID_CHECKPOINT = MODELS_DIR / "hybrid_best.pt"
LOGREG_MODEL = MODELS_DIR / "logreg_best.joblib"
XGB_MODEL = MODELS_DIR / "xgb_best.joblib"
TRAINING_METADATA = MODELS_DIR / "training_metadata.json"
EVAL_METRICS = REPORTS_DIR / "evaluation_metrics.json"
MODEL_CARD = ROOT_DIR / "model_card.md"

# MIT-BIH
PHYSIONET_DB = "mitdb"
FS = 360
ALL_RECORDS = (
    "100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 122 123 124 "
    "200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 222 223 228 230 231 232 233 234"
).split()
PACED_RECORDS = ["102", "104", "107", "217"]  # excluded per AAMI EC57
USABLE_RECORDS = [r for r in ALL_RECORDS if r not in PACED_RECORDS]

# de Chazal et al. (2004) inter-patient split
DS1_RECORDS = "101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230".split()
DS2_RECORDS = "100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234".split()

PREFERRED_LEAD = "MLII"

# AAMI EC57 classes
CLASS_NAMES = ["N", "S", "V", "F"]
CLASS_DESCRIPTIONS = {
    "N": "Normal / bundle branch block / escape",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion of ventricular and normal",
}
NUM_CLASSES = len(CLASS_NAMES)
AAMI_MAP = {s: 0 for s in "NLRej"} | {s: 1 for s in "AaJS"} | {"V": 2, "E": 2, "F": 3}
AAMI_EXCLUDED = {"/", "f", "Q"}  # paced / unclassifiable

# Signal processing
BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER = 0.5, 40.0, 4
WINDOW_BEFORE = WINDOW_AFTER = 140
WINDOW_SIZE = WINDOW_BEFORE + WINDOW_AFTER  # 280 samples, ~0.78 s
QRS_HALF_WIDTH = 20
PRE_ENERGY_SLICE = (0, 100)
POST_ENERGY_SLICE = (180, 280)
AMPLITUDE_LIMIT_MV = 5.0  # checked on the filtered, un-normalised signal

RR_LOCAL_WINDOW = 32  # beats in the causal local RR average (~30 s)
TEMPLATE_BEATS = 16  # beats in the causal per-patient median template

RR_FEATURES = ["rr_prev", "rr_next", "rr_ratio", "local_hr", "rr_local_avg", "rr_prev_norm", "rr_next_norm"]
WAVEFORM_FEATURES = ["wf_mean", "wf_std", "wf_min", "wf_max", "wf_skew", "wf_kurtosis"]
FEATURE_COLUMNS = RR_FEATURES + WAVEFORM_FEATURES + ["qrs_proxy", "pre_energy", "post_energy"]

# Training
SEED = 42
CV_FOLDS = 3
VAL_RECORD_FRACTION = 0.2
CNN_PARAMS = {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 256, "max_epochs": 50, "patience": 7}

# Final model, picked by 5-fold grouped CV on DS1 (docs/EXPERIMENTS.md, iteration 4).
# Refit epochs = median best epoch over the CV folds (14, 10, 2, 2, 7).
HYBRID_MODEL_TYPE = "hybrid_rel"
HYBRID_AUGMENT = True
HYBRID_REFIT_EPOCHS = 7
ENSEMBLE_SEEDS = [42, 7, 2024]

LOGREG_GRID = {"C": [0.01, 0.1, 1.0, 10.0]}
# Shallow trees on purpose: the first grid kept picking its smallest corner.
XGB_GRID = {"max_depth": [2, 3, 5], "n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}

MODEL_VERSION = "1.0.0"
DISCLAIMER = "Research demonstration only. Not for clinical use."
