# src/models/predict_model.py
import pandas as pd
from pgmpy.inference import VariableElimination

class ModelPredictor:
    """
    This class handles making predictions using the trained models.
    """

    def predict_decision_tree(self):
        """
        Makes predictions using the trained Decision Tree model.
        """
        pass

    def predict_neural_network(self):
        """
        Makes predictions using the trained Neural Network model.
        """
        pass

    def predict_naive_bayes(self):
        """
        Makes predictions using the trained Naive Bayes model.
        """
        pass
    @staticmethod
    def predict_bayesian_network(test_text_data, model_trainer):
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

    def predict_hidden_markov_model(self):
        """
        Makes predictions using the trained Hidden Markov Model.
        """
        pass
