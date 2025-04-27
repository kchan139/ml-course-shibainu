import os

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'processed')
EXTERNAL_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'external')

TEST_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'test')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'trained')
EXPERIMENT_DIR = os.path.join(BASE_DIR, 'models', 'experiments')

# Log and report paths
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# Other configurations
TEST_DIR = os.path.join(RAW_DATA_PATH, "all-data.csv")