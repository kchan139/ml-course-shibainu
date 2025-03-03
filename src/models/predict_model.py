# src/models/predict_model.py
import os
import pickle
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from src.config import *

class ModelPredictor:
    """
    This class handles making predictions using the trained models.
    """

    def predict_decision_tree(self):
        """
        Makes predictions using the trained Decision Tree model.
        """
        pass

    def predict_neural_network(self, test_texts):
        """
        Makes predictions using the trained BERT model.

        Args:
            test_texts (list): List of test text samples.
            
        Returns:
            Predictions: List of predicted sentiment labels.
        """
        # Tokenize the test data
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**test_encodings)
            logits = outputs.logits

        # Get predicted classes
        predictions = torch.argmax(logits, dim=-1).tolist()

        return predictions

    def predict_naive_bayes(self):
        """
        Makes predictions using the trained Naive Bayes model.
        """
        pass

    def predict_bayesian_network(self, test_text_data, model_trainer):
        """
        Args:
            test_text_data: Iterable (list, Series) of cleaned text strings.
            model_trainer: An instance of ModelTrainer that has the following attributes:
                - trained_model: The trained BayesianNetwork.
                - cv: The fitted CountVectorizer.
                - selector: The fitted SelectKBest object.
                - selected_features: List of selected feature names.
                
        Returns:
            A list of predicted labels.
        """
        # Transform the test text using the fitted CountVectorizer
        X_counts_test = model_trainer.cv.transform(test_text_data)

        # Apply the same feature selection as during training
        X_selected_test = model_trainer.selector.transform(X_counts_test)
        X_features_test = (X_selected_test > 0).astype(int)

        # Create a DataFrame with the selected feature names
        df_test = pd.DataFrame(X_features_test.toarray(), columns=model_trainer.selected_features)
        
        # Initialize the inference engine using the trained Bayesian network model.
        infer = VariableElimination(model_trainer.trained_model)
        predictions = []
        for _, row in df_test.iterrows():
            # Build the evidence dictionary from the row (convert values to int)
            evidence = {col: int(row[col]) for col in df_test.columns}

            # Query the network for the 'label' variable.
            query_result = infer.query(variables=["label"], evidence=evidence)
            
            # Pick the label with the highest probability.
            predicted_label = query_result.values.argmax()
            predictions.append(predicted_label)
            
        return predictions

    def predict_hidden_markov_model(self, test_data, model_trainer=None, model_path=None):
        """
        Makes predictions using the trained Hidden Markov Models for sentiment analysis.
        
        Args:
            test_data: Vectorized test features (sparse matrix from TF-IDF)
            model_trainer: Optional - an instance of ModelTrainer with trained HMM models
            model_path: Optional - path to the saved HMM models if model_trainer is not provided
            
        Returns:
            Predicted sentiment labels
        """
        import numpy as np
        import pickle
        import os
        from src.config import MODEL_DIR
        
        # Load models either from trainer or from file
        if model_trainer is not None and hasattr(model_trainer, 'hmm_models'):
            hmm_models = model_trainer.hmm_models
        elif model_path is not None:
            with open(model_path, "rb") as f:
                hmm_models = pickle.load(f)
        else:
            # Default path
            default_path = os.path.join(MODEL_DIR, "hmm_models.pkl")
            with open(default_path, "rb") as f:
                hmm_models = pickle.load(f)
        
        # Convert sparse matrix to dense format if needed
        if hasattr(test_data, "toarray"):
            test_data_dense = test_data.toarray()
        else:
            test_data_dense = test_data
        
        # Ensure the model dictionary is not empty
        if not hmm_models:
            raise ValueError("No HMM models were successfully trained")
        
        # Get all available sentiment labels
        sentiments = list(hmm_models.keys())
        
        # Make predictions for each sample
        predictions = []
        
        for i in range(test_data_dense.shape[0]):
            sample = test_data_dense[i:i+1]  # Get one sample
            
            # Calculate log likelihood for each sentiment model
            log_likelihoods = {}
            for sentiment, model in hmm_models.items():
                try:
                    log_likelihood = model.score(sample)
                    log_likelihoods[sentiment] = log_likelihood
                except Exception as e:
                    print(f"Error scoring sample {i} with model {sentiment}: {e}")
                    # Instead of -inf, use a very low but finite score
                    log_likelihoods[sentiment] = -1e10
            
            # Predict the sentiment with highest log likelihood
            if log_likelihoods:
                predicted_sentiment = max(log_likelihoods, key=log_likelihoods.get)
                predictions.append(predicted_sentiment)
            else:
                # Default to most common sentiment if all models fail
                most_common_sentiment = max(sentiments, key=lambda s: len(hmm_models[s].monitor_.history))
                predictions.append(most_common_sentiment)
        
        return np.array(predictions)
    

