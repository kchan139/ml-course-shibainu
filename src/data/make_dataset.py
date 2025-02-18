import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH  # Import paths from config.py

class DatasetLoader:
    """
    This class handles the processing and saving of the dataset.
    """
    def __init__(self):
        # Initialize directories from config.py
        self.raw_dir = Path(RAW_DATA_PATH)
        self.processed_dir = Path(PROCESSED_DATA_PATH)
        

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads the dataset from a local CSV file into a DataFrame.
        Args:
            file_path: Path to the local dataset file.
        Returns:
            DataFrame containing the dataset.
        """
        try:
            # Convert file_path to string to ensure compatibility with split()
            file_path = str(file_path)
            
            # Load the dataset from the local file path
            df = pd.read_csv(file_path)
            
            # Save raw data to the raw directory (using config path)
            raw_path = self.raw_dir / Path(file_path).name  # Save it with the original filename
            df.to_csv(raw_path, index=False)
            
            return df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None


    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic preprocessing of the dataset.
        """
        # Handle missing values by dropping rows with any missing values
        df = df.dropna()
        
        # Save processed data
        processed_path = self.processed_dir / "processed_moodify_dataset.csv"
        df.to_csv(processed_path, index=False)
        
        return df


    def load_processed_data(self) -> pd.DataFrame:
        """
        Loads the processed dataset if it exists.
        """
        processed_path = self.processed_dir / "processed_moodify_dataset.csv"
        if processed_path.exists():
            return pd.read_csv(processed_path)
        return None
