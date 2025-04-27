# tests/test_models.py
import pytest
from pathlib import Path
from src.config import MODEL_DIR
from src.models.predict_model import ModelPredictor

def test_decision_tree_model():
    """
    Tests the functionality of the Decision Tree model.
    """
    pass

def test_neural_network_model():
    """Tests Neural Network predictions with pre-trained model"""
    predictor = ModelPredictor()
    model_files = list(Path(MODEL_DIR).glob('*rnn*.pkl')) + list(Path(MODEL_DIR).glob('*neural*.pkl'))
    
    if not model_files:
        pytest.skip("No pre-trained Neural Network model found")

    test_headlines = [
        "Tech sector shows strong performance",
        "Regulatory challenges impact industry",
        "Innovation drives market growth"
    ]
    
    predictions = predictor.predict_neural_network(test_headlines)
    assert predictions is not None
    assert len(predictions) == 3
    for pred in predictions:
        assert 'probabilities' in pred
        assert len(pred['probabilities']) == 3
        assert abs(sum(pred['probabilities'].values()) - 1.0) < 0.0001

def test_naive_bayes_model():
    """Tests Naive Bayes predictions with pre-trained model"""
    predictor = ModelPredictor()
    model_path = Path(MODEL_DIR) / 'naive_bayes_model.pkl'
    
    if not model_path.exists():
        pytest.skip("No pre-trained Naive Bayes model found")

    test_headlines = [
        "Company reports strong earnings growth",
        "Market volatility concerns rise",
        "New product launch delayed"
    ]
    
    predictions = predictor.predict_naive_bayes(test_headlines)
    assert predictions is not None
    assert len(predictions) == 3
    for pred in predictions:
        assert 'sentiment' in pred
        assert pred['sentiment'] in ['negative', 'neutral', 'positive']

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
