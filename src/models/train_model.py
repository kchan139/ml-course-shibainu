# src/models/train_model.py
import os
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from pgmpy.models import BayesianNetwork  # or BayesianModel (deprecated, use BayesianNetwork)
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from src.config import MODEL_DIR
import matplotlib.pyplot as plt
import os
import pickle
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
from datasets import Dataset
from src.config import *
from gensim.models import Word2Vec
from sklearn.preprocessing import KBinsDiscretizer
import logging  
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def _compute_class_weights(self, y):
    """
    Compute class weights to handle potential class imbalance.
    
    Args:
        y: Training labels (encoded)
    
    Returns:
        Dictionary of class weights
    """
    unique_classes = np.unique(y)
    class_counts = np.bincount(y)
    total_samples = len(y)
    
    # Compute weights inversely proportional to class frequencies
    weights = {
        cls: total_samples / (len(unique_classes) * count) 
        for cls, count in zip(unique_classes, class_counts)
    }
    
    return weights

class ModelTrainer:
    """
    This class is responsible for training different models using the processed features.
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def train_decision_tree(self, vectorized_data, labels):
        """
        Trains the Decision Tree model.

        Args:
            vectorized_data: Sparse matrix or array output from a vectorizer.
            labels: Array-like structure with the numeric label for each sample.
        """
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(vectorized_data, labels, test_size=0.2, random_state=42)
        
        # Initialize and train the Decision Tree classifier
        self.decision_tree_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
        self.decision_tree_model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.decision_tree_model.predict(X_test)
        print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred))

        return self.decision_tree_model

    def train_neural_network(self, X_train, y_train, X_test=None, y_test=None, epochs=3, batch_size=8):
        """
        Trains a neural network using BERT for text classification.
        
        Args:
            X_train (list): List of training text samples.
            y_train (list): List of training labels.
            X_test (list): Optional list of test text samples.
            y_test (list): Optional test labels.
            epochs (int): Number of epochs.
            batch_size (int): Batch size for training.
        """
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Tokenize the training data
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=256)
        
        if X_test is not None:
            test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length=256)
        else:
            test_encodings = None

        # Convert the training and test data to datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': y_train
        })
        
        if X_test is not None:
            test_dataset = Dataset.from_dict({
                'input_ids': test_encodings['input_ids'],
                'attention_mask': test_encodings['attention_mask'],
                'labels': y_test
            })
        else:
            test_dataset = None

        # Load pre-trained BERT model for sequence classification
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(y_train)))

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=epochs,         # number of training epochs
            per_device_train_batch_size=batch_size,  # batch size for training
            per_device_eval_batch_size=batch_size,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            evaluation_strategy="epoch"      # evaluation strategy
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,                     # the model to train
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset            # evaluation dataset
        )

        # Train the model
        trainer.train()

        return self.model


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
        print(best_structure.edges())

        # Save the trained model.
        save_path = os.path.join(MODEL_DIR, 'bayesian_network_model.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.trained_model, f)
        
        return self.trained_model

    def train_bayesian_network_W2V(self, text_column, label_column, vector_size=70, window=5, min_count=1, n_bins=35):
        """
        Trains a Bayesian network using word2vec sentence embeddings.
        
        Args:
            text_column: A list (or Series) containing the cleaned text.
            label_column: A list (or Series) containing the labels.
            vector_size: Dimensionality of the word vectors.
            window: Maximum distance between the current and predicted word within a sentence.
            min_count: Ignores all words with total frequency lower than this.
            n_bins: Number of bins to discretize each feature.
        
        Returns:
            A trained BayesianNetwork instance.
        """
        # 1. Tokenize the text: split each sentence into words.
        sentences = [text.split() for text in text_column]
        
        logging.getLogger("gensim").setLevel(logging.WARNING)
        # 2. Train a Word2Vec model on these tokenized sentences.
        self.w2v_model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
        
        # Function to compute average embedding for a sentence.
        def sentence_embedding(sentence, model):
            # Only include words that exist in the model's vocabulary.
            embeddings = [model.wv[word] for word in sentence if word in model.wv]
            if embeddings:
                return np.mean(embeddings, axis=0)
            else:
                # If no words in the sentence are in the vocabulary, return a zero vector.
                return np.zeros(model.vector_size)
        
        # 3. Compute sentence embeddings for each text sample.
        X_w2v = np.array([sentence_embedding(sent, self.w2v_model) for sent in sentences])

        # 4. Discretize the continuous embeddings.
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        X_discrete = self.discretizer.fit_transform(X_w2v)
        
        # Build a DataFrame from the discretized features.
        feature_names = [f"feat_{i}" for i in range(X_discrete.shape[1])]
        df_features = pd.DataFrame(X_discrete, columns=feature_names)
        
        # Ensure the labels are in a suitable format.
        if not isinstance(label_column, pd.Series):
            label_column = pd.Series(label_column)
        df_features['label'] = label_column.values
        
        # 5. Learn the Bayesian Network structure and parameters.
        hc = HillClimbSearch(df_features)
        best_structure = hc.estimate(scoring_method=BicScore(df_features))
        self.trained_model = BayesianNetwork(best_structure.edges())
        self.trained_model.fit(df_features, estimator=BayesianEstimator, prior_type='BDeu')
        
        # 6. Save the trained model using a relative path.
        save_path = os.path.join(MODEL_DIR, 'bayesian_network_model_W2V.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.trained_model, f)
            
        return self.trained_model

    def train_hidden_markov_model(self, X_train, y_train, n_components=2, random_state=42):
        """
        Trains a Hidden Markov Model (HMM) using the preprocessed data for sentiment analysis.
        
        Args:
            X_train: Vectorized training features (sparse matrix from TF-IDF)
            y_train: Training labels (sentiment)
            n_components: Number of hidden states in the HMM (default=2)
            random_state: Random seed for reproducibility
            
        Returns:
            A trained HMM instance
        """
        
        # Convert sparse matrix to dense format if needed
        if hasattr(X_train, "toarray"):
            X_train_dense = X_train.toarray()
        else:
            X_train_dense = X_train
        
        # Apply StandardScaler to normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_dense)

        # Determine the number of distinct sentiments/classes
        n_labels = len(np.unique(y_train))
        
        # Create and train separate HMM for each sentiment class
        hmm_models = {}
        
        for sentiment in range(n_labels):
            # Filter data for current sentiment
            X_sentiment = X_train_scaled[y_train == sentiment]
            
            if len(X_sentiment) < n_components:
                print(f"Warning: Not enough samples ({len(X_sentiment)}) for sentiment {sentiment} with {n_components} components")
                continue
                
            # Train HMM for this sentiment
            sentiment_model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=200,  # Increased iterations
                random_state=random_state
            )
            
            # Fit the model
            try:
                sentiment_model.fit(X_sentiment)
                hmm_models[sentiment] = sentiment_model
                print(f"Successfully trained HMM for sentiment {sentiment} with {len(X_sentiment)} samples")
            except Exception as e:
                print(f"Error training HMM for sentiment {sentiment}: {e}")
        
        # Save the models
        file_path = os.path.join(MODEL_DIR, "hmm_models.pkl")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
        with open(file_path, "wb") as f:
            pickle.dump(hmm_models, f)
        print(f"HMM models saved at: {file_path}")
        
        # Store the models in the instance for later use
        self.hmm_models = hmm_models
        
        return hmm_models