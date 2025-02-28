# src/models/train_model.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator

class ModelTrainer:
    """
    This class is responsible for training different models using the processed features.
    """

    def train_decision_tree(self):
        """
        Trains the Decision Tree model using the processed features. iaeiia
        """
        pass

    def train_neural_network(self):
        """
        Trains the Neural Network (CNN-BiLSTM) model.
        """
        pass

    def train_naive_bayes(self):
        """
        Trains the Naive Bayes model using frequency analysis.
        """
        pass

    def train_bayesian_network(self, vectorized_data, label):
        """
        Args:
            vectorized_data: Sparse matrix or array output from a vectorizer.
            label: Array-like structure with the numeric label for each sample.

        Returns:
            A trained BayesianModel instance.
        """
        # Convert sparse matrix to dense array if applicable.
        # if hasattr(vectorized_data, "toarray"):
        X_dense = vectorized_data.toarray()
        # else:
        #     X_dense = vectorized_data

        # Reduce dimensionality using PCA 
        self.pca = PCA(n_components=90)
        X_reduced = self.pca.fit_transform(X_dense)

        # Discretize the PCA-reduced features into categorical bins
        self.discretizer = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile')
        X_discrete = self.discretizer.fit_transform(X_reduced)

        # Create a DataFrame with the discretized features
        df_features = pd.DataFrame(X_discrete, columns=[f"feat_{i}" for i in range(X_discrete.shape[1])])
        # Append the label column (assumed to be numeric)
        df_features["label"] = label

        # Learn the Bayesian Network structure using Hill-Climb Search with the BIC score
        hc = HillClimbSearch(df_features)
        best_structure = hc.estimate(scoring_method=BicScore(df_features))

        # Create the BayesianModel using the learned structure (edges)
        self.trained_model = BayesianNetwork(best_structure.edges())

        # Step 5: Fit the model parameters using a Bayesian estimator with a BDeu prior
        self.trained_model.fit(df_features, estimator=BayesianEstimator, prior_type='BDeu')

        return self.trained_model

    def train_hidden_markov_model(self):
        """
        Trains the Hidden Markov Model for emotion prediction in songs.
        """
        pass
