# tests/test_data.py
import pytest
import pandas as pd
from src.data.make_dataset import DatasetLoader

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



def test_data_cleaning():
    """
    Verifies the data cleaning process works as expected.
    """
    pass


def test_data_split():
    """
    Ensures data is correctly split into train, validation, and test sets.
    """
    pass


def test_vectorization():
    """
    Validates that the lyrics are correctly transformed into numerical representations.
    """
    pass
