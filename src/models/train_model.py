# src/models/train_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from pgmpy.models import BayesianNetwork  # or BayesianModel (deprecated, use BayesianNetwork)
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

    def train_bayesian_network(self, text_column, label_column, k_features=90):
        """
        Trains a Bayesian network using features extracted from text.
        
        Args:
            text_column: The list containing the cleaned text.
            label_column: The list containing the labels.
            k_features: Number of top n-gram features to select.
        
        Returns:
            A trained BayesianNetwork instance.
        """
        
        # Extract n-gram counts (unigrams and bigrams).
        self.cv = CountVectorizer(ngram_range=(1,2), stop_words='english')
        X_counts = self.cv.fit_transform(text_column)
        
        # Select the top k features using a chi-square test.
        y = label_column
        self.selector = SelectKBest(chi2, k=k_features)
        X_selected = self.selector.fit_transform(X_counts, y)
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = [self.cv.get_feature_names_out()[i] for i in selected_indices]
        
        # Build a feature DataFrame.
        X_features = (X_selected > 0).astype(int)
        df_features = pd.DataFrame(X_features.toarray(), columns=self.selected_features)
        df_features['label'] = y.values
        
        # Learn the Bayesian Network structure and parameters.
        hc = HillClimbSearch(df_features)
        best_structure = hc.estimate(scoring_method=BicScore(df_features))
        self.trained_model = BayesianNetwork(best_structure.edges())
        self.trained_model.fit(df_features, estimator=BayesianEstimator, prior_type='BDeu')
        
        return self.trained_model

    def train_hidden_markov_model(self):
        """
        Trains the Hidden Markov Model for emotion prediction in songs.
        """
        pass
