# tests/test_data.py
import pytest
import pandas as pd
import os
from src.config import *
from src.data.make_dataset import DatasetLoader
from src.data.preprocess import DataPreprocessor

def test_load_data(tmp_path):
    """
    Tests whether the dataset is properly loaded from a local file using temporary directories.
    Verifies both successful loading and error handling.
    """
    # Setup: Create a sample DataFrame and write it to a temporary CSV file
    data = {
        'Sentiment': [
            'neutral', 'negative', 'positive'
        ],
        'News Headline': [
            "According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .",
            "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .", 
            "According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales ."
        ],
    }
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a temporary CSV file
    temp_file = tmp_path / "test_data.csv"
    df.to_csv(temp_file, index=False)

    # Initialize the DatasetLoader
    loader = DatasetLoader()

    # Test: Load the data using the DatasetLoader
    loaded_df = loader.load_data(temp_file)

    # Assertions
    assert loaded_df is not None, "Failed to load the dataset"
    assert isinstance(loaded_df, pd.DataFrame), "Loaded data is not a DataFrame"
    pd.testing.assert_frame_equal(df, loaded_df), "The loaded data does not match the original data"

    # Test: Handle loading of a non-existing file
    invalid_file = tmp_path / "non_existing_file.csv"
    invalid_data = loader.load_data(invalid_file)
    assert invalid_data is None, "The dataset loader should return None for a non-existing file"

    # Clean up
    temp_file.unlink()


def test_process_data():
    """
    pass
    Tests the data processing method (removing duplicates, handling missing values).
    """
    pass
    # Setup: Create a sample DataFrame with duplicates and missing values
    # Setup: Create a sample DataFrame and write it to a temporary CSV file
    data = {
        'Sentiment': [
            'neutral', 'negative', 'positive', 'positive'
        ],
        'News Headline': [
            "According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .",
            "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .", 
            "According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales .",
            None
        ],
    }
    df = pd.DataFrame(data)

    # Initialize the DatasetLoader
    loader = DatasetLoader()

    # Test: Process the data
    processed_df = loader.process_data(df)

    # Assertions
    assert processed_df is not None, "Failed to process the dataset"

    # Check for removed duplicates (rows with '2500' should be removed)
    assert len(processed_df) == 3, f"Expected 3 rows, but got {len(processed_df)}"
    
    # Check that missing values are dropped
    assert processed_df['News Headline'].isnull().sum() == 0, "There are still missing values in 'News Headline'"



def test_load_processed_data():
    """
    Tests whether the processed data can be loaded correctly.
    """
    pass
    # Setup: Create a sample DataFrame to simulate the processed data
    data = {
        'Sentiment': [
            'neutral', 'negative', 'positive'
        ],
        'News Headline': [
            "According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .",
            "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .", 
            "According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales ."
        ],
    }
    df = pd.DataFrame(data)
    
    # Initialize the DatasetLoader
    loader = DatasetLoader()

    # Save the processed data to the processed directory
    processed_path = loader.processed_dir / "test_data.csv"
    df.to_csv(processed_path, index=False)

    # Test: Load the processed data
    loaded_processed_df = loader.load_processed_data()

    # Assertions
    assert loaded_processed_df is not None, "Failed to load the processed dataset"
    assert isinstance(loaded_processed_df, pd.DataFrame), "The loaded data is not a DataFrame"
    
    # Check if the loaded processed data matches the original processed data
    pd.testing.assert_frame_equal(df, loaded_processed_df), "The loaded processed data does not match the saved data"
    
    # Clean up
    processed_path.unlink()



def create_sample_csv(tmp_path):
    """
    Helper function to create a sample CSV file for testing.
    Returns the file path and the original DataFrame.
    """
    # Build the path relative to the current file
    file_path = TEST_DIR

    # Read the CSV
    df = pd.read_csv(file_path)

    loader = DatasetLoader()
    processed_df = loader.process_data(df)

    file_path = tmp_path / "sample.csv"
    processed_df.to_csv(file_path, index=False)
    return file_path, processed_df

def test_data_cleaning(tmp_path):
    """
    Verifies that the data cleaning process works as expected:
      - Creates a 'clean_text' column.
      - Encodes sentiment labels into 'sentiment_encoded'.
      - Checks that text is lowercased and punctuation is removed.
    """
    file_path, original_df = create_sample_csv(tmp_path)
    preprocessor = DataPreprocessor(str(file_path))
    preprocessor.clean_data()
    processed_df = preprocessor.get_processed_dataframe()
    
    # Check if the 'clean_text' column exists
    assert "clean_text" in processed_df.columns, "Missing 'clean_text' column after cleaning."
    
    # Check that the cleaning has been applied (e.g., lowercase, punctuation removed)
    sample_cleaned = processed_df["clean_text"].iloc[0]
    expected_cleaned = "according gran company plan move production russia although company growing"
    assert sample_cleaned == expected_cleaned, f"Expected '{expected_cleaned}' but got '{sample_cleaned}'"
    
    # Check if the 'sentiment_encoded' column exists and has three unique values
    assert "sentiment_encoded" in processed_df.columns, "Missing 'sentiment_encoded' column after cleaning."
    unique_labels = processed_df["sentiment_encoded"].unique()
    # Since label encoding is arbitrary, we expect three unique numeric labels.
    assert len(unique_labels) == 3, "Sentiment labels were not encoded correctly."

def test_data_split(tmp_path):
    """
    Ensures that the dataset is correctly split into training, validation, and test sets.
    Checks that the split sizes roughly correspond to the specified test and validation sizes.
    """
    file_path, _ = create_sample_csv(tmp_path)
    preprocessor = DataPreprocessor(str(file_path))
    preprocessor.clean_data()
    total = len(preprocessor.df)
    preprocessor.split_data(test_size=0.25, val_size=0.2, random_state=42)
    
    # Ensure that all data is accounted for in the splits
    total_split = len(preprocessor.X_train) + len(preprocessor.X_val) + len(preprocessor.X_test)
    assert total == total_split, "The sum of splits does not equal the total number of samples."
    
    # Check test set size (expected about 25% of total)
    expected_test_size = int(total * 0.25)
    assert abs(len(preprocessor.X_test) - expected_test_size) <= 1, "Test set size is not as expected."

def test_vectorization(tmp_path):
    """
    Validates that the vectorization process produces TF-IDF matrices
    with the correct dimensions based on the data splits.
    """
    file_path, _ = create_sample_csv(tmp_path)
    preprocessor = DataPreprocessor(str(file_path))
    preprocessor.clean_data()
    preprocessor.split_data(test_size=0.3, val_size=0.2, random_state=42)
    
    # Vectorize the cleaned text
    (X_train_vec, X_val_vec, X_test_vec), vectorizer = preprocessor.vectorize_text()
    
    # Verify the vectorizer is initialized
    assert vectorizer is not None, "TF-IDF vectorizer was not initialized."
    
    # Get the number of features (should not exceed 5000 as specified)
    num_features = len(vectorizer.get_feature_names_out())
    assert num_features <= 5000, "Number of TF-IDF features exceeds the specified limit."
    
    # Check that the number of rows in each matrix matches the respective split sizes
    assert X_train_vec.shape[0] == len(preprocessor.X_train), "Mismatch in training set size after vectorization."
    assert X_val_vec.shape[0] == len(preprocessor.X_val), "Mismatch in validation set size after vectorization."
    assert X_test_vec.shape[0] == len(preprocessor.X_test), "Mismatch in test set size after vectorization."
