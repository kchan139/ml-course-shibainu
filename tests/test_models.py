# tests/test_models.py
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.config import *
from src.data.preprocess import DataPreprocessor
from src.models.train_model import ModelTrainer
from src.models.predict_model import ModelPredictor
import os
import pandas as pd

def test_decision_tree_model():
    """
    Tests the functionality of the Decision Tree model.
    """
    pass

def test_neural_network_model():
    """
    Tests the functionality of the Neural Network model.
    """
    # Load and preprocess the dataset
    file_path = TEST_DIR
    preprocessor = DataPreprocessor(file_path)
    preprocessor.clean_data()
    preprocessor.split_data(test_size=0.2, random_state=42)  # 80% train, 20% test

    # Train the neural network model with raw text data
    trainer = ModelTrainer()
    _ = trainer.train_neural_network(preprocessor.X_train, preprocessor.y_train, 
                                    preprocessor.X_test, preprocessor.y_test, 
                                    epochs=5, batch_size=32)

    # Make predictions using the trained model
    _, predictions = ModelPredictor().predict_neural_network(preprocessor.X_test, trainer)
    # Evaluate the model
    accuracy = accuracy_score(preprocessor.y_test, predictions)
    print(f"Accuracy: {accuracy}")

    # Classification report
    print("Classification Report:")
    print(classification_report(preprocessor.y_test, predictions))

    # Confusion matrix
    cm = confusion_matrix(preprocessor.y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Assertions for testing
    assert accuracy > 0.5, f"Accuracy is below the expected threshold. Got {accuracy:.2f}"

    # Additional checks like classification report and confusion matrix can be added if necessary.
    # For example, checking if confusion matrix is of the expected shape
    assert cm.shape == (3, 3), "Confusion matrix has incorrect dimensions. It should be 3x3."

def test_naive_bayes_model():
    """
    Tests the functionality of the Naive Bayes model.
    """
    pass

def test_bayesian_network_model():
    """
    Tests the functionality of the Bayesian Network model.
    """
    pass

def test_hidden_markov_model():
    """
    Tests the functionality of the Hidden Markov Model.
    """
    pass
