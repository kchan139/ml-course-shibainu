import os
from pathlib import Path

# Base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'processed')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'trained_models')
EXPERIMENT_DIR = os.path.join(BASE_DIR, 'models', 'experiments')

# MLflow configuration
MLFLOW_TRACKING_URI = f"file:///{str(Path(EXPERIMENT_DIR).resolve()).replace('\\', '/')}"
MLFLOW_ARTIFACT_PATH = str(Path(MODEL_DIR).resolve())