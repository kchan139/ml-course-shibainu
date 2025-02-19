import pandas as pd
from pathlib import Path
from src.config import *
from src.data.preprocess import DataPreprocessor

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
        Process the dataset using the DataPreprocessor.
        This involves cleaning, encoding sentiments, and splitting the data.
        """
        # Initialize the DataPreprocessor
        preprocessor = DataPreprocessor("")  # File path is not necessary for processing the DataFrame directly
        preprocessor.df = df  # Set the DataFrame to the preprocessor
        preprocessor.clean_data()  # Clean the data (remove punctuation, lowercase, etc.)

        # Get the processed DataFrame
        processed_df = preprocessor.get_processed_dataframe()

        # Return the processed DataFrame
        return processed_df


    def load_processed_data(self) -> pd.DataFrame:
        """
        Load the processed dataset using the DataPreprocessor.
        """
        # Use DataPreprocessor to load and process the data if it exists
        preprocessor = DataPreprocessor("")  # File path is not necessary for loading
        processed_df = preprocessor.get_processed_dataframe()
        
        # If the processed data doesn't exist yet, return None
        if processed_df is None:
            return None
        
        return processed_df
