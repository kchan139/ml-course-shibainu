# src/models/train_model.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Conv1D, MaxPooling1D, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from ..features.build_features import FeatureExtractor

class ModelTrainer:
    """
    This class is responsible for training different models using the processed features.
    """

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

    def train_neural_network(self, X_train_texts, y_train, X_test_texts=None, y_test=None, epochs=10, batch_size=32):
        """
        Trains a neural network model using GloVe embeddings for text classification.
        
        Args:
            X_train_texts: List of training text samples
            y_train: Training labels
            X_test_texts: Optional list of test text samples
            y_test: Optional test labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        
        # Build vocabulary
        _, vocab_size = feature_extractor.fit_vocabulary(X_train_texts)
        
        # Convert texts to sequences and pad
        X_train_seq = feature_extractor.texts_to_sequences(X_train_texts)
        X_train_padded = feature_extractor.pad_sequences(X_train_seq)
        
        if X_test_texts is not None:
            X_test_seq = feature_extractor.texts_to_sequences(X_test_texts)
            X_test_padded = feature_extractor.pad_sequences(X_test_seq)
        else:
            X_test_padded = None
        
        # Load GloVe embeddings
        embedding_matrix = feature_extractor.load_glove_embeddings()
        max_length = feature_extractor.max_length
        
        # Encode labels if they're not already encoded
        if not isinstance(y_train[0], (int, np.integer)):
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            if y_test is not None:
                y_test = self.label_encoder.transform(y_test)

        # Convert labels to one-hot encoding
        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes) if y_test is not None else None
        
        # Define model with embedding layer
        self.nn_model = Sequential([
            Embedding(input_dim=vocab_size, 
                    output_dim=feature_extractor.embedding_dim, 
                    weights=[embedding_matrix],
                    input_length=max_length,
                    trainable=False),
            LSTM(128, kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))
        ])


        # Compile the model
        self.nn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Define early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        # Train the model
        if X_test_padded is not None and y_test is not None:
            history = self.nn_model.fit(
                X_train_padded, y_train_cat,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_padded, y_test_cat),
                callbacks=[early_stopping]
            )
        else:
            history = self.nn_model.fit(
                X_train_padded, y_train_cat,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,  # Use 20% of the training data for validation
                callbacks=[early_stopping]
            )

        # Store history for external analysis
        self.training_history = history.history  # Save metrics dictionary
        self.training_epochs = range(1, len(history.history['accuracy']) + 1)

        return self.nn_model


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
