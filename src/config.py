from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
PIPELINE_DIR = BASE_DIR / "pipelines"

# Files
RAW_DATA_FILE = RAW_DATA_DIR / "Student Mental health.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "student_mental_health_processed.csv"

MODEL_FILE = OUTPUT_DIR / "student_mental_health_model.joblib"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"

# MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Student_Mental_Health"

REGISTERED_MODEL_NAME = "StudentMentalHealthModel"

# Target
TARGET_COLUMN = "Depression"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def create_directories():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)