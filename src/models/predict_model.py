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
    def predict_bayesian_network(test_vectorized_data, model_trainer):
        """
        Makes predictions using the trained Bayesian Network model stored in the provided ModelTrainer instance.
        
        Args:
            test_vectorized_data: Sparse matrix or array of test data from a vectorizer.
            model_trainer: An instance of ModelTrainer that has the following attributes:
                - trained_model: The trained BayesianModel.
                - pca: The fitted PCA object used during training.
                - discretizer: The fitted KBinsDiscretizer used during training.
                
        Returns:
            A list of predicted labels.
        """
        # Convert test_vectorized_data to a dense array if needed.
        # if hasattr(test_vectorized_data, "toarray"):
        X_dense_test = test_vectorized_data.toarray()
        # else:
        #     X_dense_test = test_vectorized_data

        # Transform the test data using the PCA object from the model_trainer.
        X_reduced_test = model_trainer.pca.transform(X_dense_test)

        # Discretize the PCA-transformed test data using the discretizer from the model_trainer.
        X_discrete_test = model_trainer.discretizer.transform(X_reduced_test)

        # Create a DataFrame from the discretized test features.
        df_test = pd.DataFrame(X_discrete_test, columns=[f"feat_{i}" for i in range(X_discrete_test.shape[1])])

        # Initialize the inference engine with the trained Bayesian network model.
        infer = VariableElimination(model_trainer.trained_model)

        predictions = []
        for _, row in df_test.iterrows():
            # Build the evidence dictionary for the current sample.
            evidence = {col: int(row[col]) for col in df_test.columns}
            # Query the model for the 'label' variable.
            query_result = infer.query(variables=["label"], evidence=evidence)
            # Determine the predicted label by taking the argmax of the probability distribution.
            predicted_label = query_result.values.argmax()
            predictions.append(predicted_label)
            
        return predictions

    def predict_hidden_markov_model(self):
        """
        Makes predictions using the trained Hidden Markov Model.
        """
        pass
