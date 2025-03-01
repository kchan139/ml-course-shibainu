# src/models/train_model.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class ModelTrainer:
    """
    This class is responsible for training different models using the processed features.
    """

    def train_decision_tree(self):
        """
        Trains the Decision Tree model using the processed features. iaeiia
        """
        pass

    def train_neural_network(self, X_train_vec, y_train, X_test_vec=None, y_test=None, epochs=10, batch_size=32):
        """
        Trains a neural network model for sentiment classification.
        
        Args:
            X_train_vec: Vectorized training features (sparse matrix from TF-IDF)
            y_train: Training labels
            X_test_vec: Optional vectorized test features
            y_test: Optional test labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Trained neural network model
        """
        
        
        # Convert sparse matrix to dense numpy array if needed
        if hasattr(X_train_vec, "toarray"):
            X_train_dense = X_train_vec.toarray()
        else:
            X_train_dense = X_train_vec
            
        if X_test_vec is not None and hasattr(X_test_vec, "toarray"):
            X_test_dense = X_test_vec.toarray()
        else:
            X_test_dense = X_test_vec
        
        # Encode labels if they're not already encoded
        if not isinstance(y_train[0], (int, np.integer)):
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            if y_test is not None:
                y_test = self.label_encoder.transform(y_test)
        
        # Convert labels to one-hot encoding for multi-class classification
        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        if y_test is not None:
            y_test_cat = to_categorical(y_test, num_classes)
        
        # Define the model architecture
        self.nn_model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train_dense.shape[1],)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        self.nn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Define early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train the model
        if X_test_dense is not None and y_test is not None:
            history = self.nn_model.fit(
                X_train_dense, y_train_cat,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_dense, y_test_cat),
                callbacks=[early_stopping]
            )
            
            # Plot training history if testing data was provided
            plt.figure(figsize=(12, 4))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        else:
            self.nn_model.fit(
                X_train_dense, y_train_cat,
                epochs=epochs,
                batch_size=batch_size
            )
        
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
