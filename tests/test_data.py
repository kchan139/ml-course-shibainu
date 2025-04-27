# tests/test_data.py
import pytest
import pandas as pd
import numpy as np
from src.data.make_dataset import DatasetLoader
from src.data.preprocess import DataPreprocessor
from src.config import TEST_DATA_PATH

# Sample data for testing
SAMPLE_DATA = {
    'Sentiment': ['neutral', 'negative', 'positive', 'positive'],
    'News Headline': [
        "Sample headline 1",
        "Sample headline 2",
        "Sample headline 3",
        np.nan  # include missing value
    ]
}

def test_load_data():
    """Test dataset loading"""
    loader = DatasetLoader(TEST_DATA_PATH)
    
    test_df = pd.DataFrame(SAMPLE_DATA)
    
    loaded_df = loader.process_data(test_df.copy())
    
    assert loaded_df is not None
    assert len(loaded_df) == 3  # Should drop 1 row with NaN
    assert loaded_df.equals(test_df.dropna().reset_index(drop=True))

def test_process_data():
    """Test data processing"""
    loader = DatasetLoader(TEST_DATA_PATH)
    
    test_df = pd.DataFrame(SAMPLE_DATA)
    
    processed_df = loader.process_data(test_df.copy())
    
    # Verify processing results
    assert processed_df is not None
    assert processed_df['News Headline'].isna().sum() == 0
    assert len(processed_df) == 3

def test_data_cleaning():
    """Test data cleaning pipeline"""
    preprocessor = DataPreprocessor("")
    preprocessor.df = pd.DataFrame(SAMPLE_DATA)
    
    # Run cleaning pipeline
    preprocessor.clean_data()
    processed_df = preprocessor.get_processed_dataframe()
    
    # Verify cleaning results
    assert 'clean_text' in processed_df.columns
    assert 'sentiment_encoded' in processed_df.columns
    assert processed_df['clean_text'].str.islower().all()
    assert processed_df['sentiment_encoded'].nunique() == 3

SAMPLE_DATA_BALANCED = {
    'Sentiment': [
        'neutral', 'negative', 'positive', 'positive',
        'neutral', 'negative'
    ],
    'News Headline': [
        "Sample headline 1",
        "Sample headline 2",
        "Sample headline 3",
        "Sample headline 4",
        "Sample headline 5",
        "Sample headline 6"
    ]
}

def test_data_split():
    """Test data splitting functionality"""
    preprocessor = DataPreprocessor("")
    preprocessor.df = pd.DataFrame(SAMPLE_DATA_BALANCED)
    preprocessor.clean_data()
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=0.5)
    
    assert len(X_train) + len(X_test) == len(preprocessor.df)
    assert len(y_train) + len(y_test) == len(preprocessor.df)
    assert 0.4 < len(X_test)/len(preprocessor.df) < 0.6

def test_vectorization():
    """Test TF-IDF vectorization"""
    preprocessor = DataPreprocessor("")
    preprocessor.df = pd.DataFrame(SAMPLE_DATA_BALANCED)
    preprocessor.clean_data()
    preprocessor.split_data(test_size=0.5)
    
    (X_train_vec, X_test_vec), vectorizer = preprocessor.vectorize_text()
    
    assert vectorizer is not None
    assert X_train_vec.shape[0] == len(preprocessor.X_train)
    assert X_test_vec.shape[0] == len(preprocessor.X_test)
    assert X_train_vec.shape[1] == X_test_vec.shape[1]