from pathlib import Path

SEED = 42
MODEL_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "telco_customer_churn.csv"
MODELS_DIR = ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "model.pt"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
TARGET = "churn"
