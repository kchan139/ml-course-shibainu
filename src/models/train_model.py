# src/models/train_model.py
import os
import pickle
import numpy as np
import pandas as pd
from hmmlearn import hmm
from datetime import datetime

from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import compute_class_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

from pgmpy.models import BayesianNetwork  # or BayesianModel (deprecated, use BayesianNetwork)
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator

from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense, Dropout, GlobalMaxPooling1D, Embedding, Bidirectional, LSTM
)

from src.config import *
from src.data.preprocess import DataPreprocessor
from gensim.models import Word2Vec
from sklearn.preprocessing import KBinsDiscretizer
import logging


class ModelTrainer:
    """
    This class is responsible for training different models using the processed features.
    """

    def train_decision_tree(self, X_train, y_train):
        """
        Trains a Decision Tree model.

        Args:
            X_train: Feature matrix (e.g., sparse matrix or array) for training.
            y_train: Array-like structure with the numeric labels for each training sample.

        Returns:
            A trained DecisionTreeClassifier instance.
        """
        
        # Define a parameter grid for hyperparameter tuning
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_leaf': [1, 2, 4, 8, 16],
            'criterion': ['entropy', 'gini']
        }
        
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        self.decision_tree_model = grid_search.best_estimator_
        
        # Save the trained model to the relative directory.
        save_path = os.path.join(MODEL_DIR, 'decision_tree_model.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.decision_tree_model, f)
            
        return self.decision_tree_model

    def train_neural_network(self, preprocessor=None, file_path=None, max_words=3000, 
                    embedding_dim=32, max_len=30, epochs=20, batch_size=16):
        """
        Trains a balanced RNN model for news headline sentiment classification,
        with specific handling for class imbalance.
        
        Args:
            preprocessor: Optional DataPreprocessor object with preprocessed data
            file_path: Path to the processed data file if preprocessor is None
            max_words: Maximum number of words for tokenization
            embedding_dim: Dimension for the embedding layer
            max_len: Maximum length of sequences for headlines
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Trained RNN model
        """
        print("Starting RNN model training...")
        
        # Prepare data
        if preprocessor is None and file_path is not None:
            preprocessor = DataPreprocessor(file_path)
            preprocessor.clean_data()
            preprocessor.split_data()
        
        if preprocessor is None:
            default_path = os.path.join(PROCESSED_DATA_PATH, "processed_dataset.csv")
            preprocessor = DataPreprocessor(default_path)
            preprocessor.clean_data()
            preprocessor.split_data()
        
        # Access data
        X_train = preprocessor.X_train
        X_test = preprocessor.X_test
        y_train = preprocessor.y_train
        y_test = preprocessor.y_test

        # Calculate class weights to handle imbalance
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        print(f"Class weights for balancing: {class_weights}")

        # Convert to categorical format
        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        # Tokenize
        tokenizer = Tokenizer(num_words=max_words, lower=True)
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

        # Define model architecture for better class separation
        model = Sequential([
            # Embedding
            Embedding(max_words, embedding_dim),
            
            # Bidirectional LSTM for better context capture
            Bidirectional(LSTM(32, return_sequences=True)),
            
            # Global max pooling to capture the most important features
            GlobalMaxPooling1D(),
            
            # Dropout for regularization
            Dropout(0.3),
            
            # Hidden layer for better discrimination between classes
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.3),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])

        # Compile with focal loss to focus more on difficult examples
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        # Add callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.0001
        )

        # Train with class weights
        history = model.fit(
            X_train_pad, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights  # Apply class weights here
        )
        
        # Custom threshold function to prevent bias toward majority class
        def custom_prediction(probs, threshold=0.3):
            predictions = []
            for prob in probs:
                # If no class exceeds threshold, pick highest
                if not any(p > threshold for p in prob):
                    predictions.append(np.argmax(prob))
                # Otherwise use custom logic to favor minority classes slightly
                elif prob[0] > threshold*0.9:  # Lower threshold for negative class
                    predictions.append(0)
                elif prob[2] > threshold*0.9:  # Lower threshold for positive class
                    predictions.append(2)
                else:
                    predictions.append(np.argmax(prob))
            return np.array(predictions)
        
        # Evaluate with custom threshold
        y_pred_prob = model.predict(X_test_pad)
        y_pred = custom_prediction(y_pred_prob)
        
        # Convert one-hot encoded test data back to class indices for evaluation
        y_test_indices = np.argmax(y_test_cat, axis=1)
        
        # Print evaluation metrics
        accuracy = accuracy_score(y_test_indices, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test_indices, y_pred))
        
        # Check class distribution in predictions
        print("Prediction class distribution:")
        unique, counts = np.unique(y_pred, return_counts=True)
        for class_idx, count in zip(unique, counts):
            print(f"Class {class_idx}: {count} predictions")
        
        # Save the model with custom threshold logic
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"rnn_{timestamp}.pkl"
        model_path = os.path.join(EXPERIMENT_DIR, model_filename)
        
        model_data = {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': preprocessor.label_encoder,
            'max_len': max_len,
            'metrics': {
                'accuracy': accuracy
            },
            'custom_threshold': 0.3  # Save threshold for prediction
        }
        
        with open(model_path, 'wb') as file:
            pickle.dump(model_data, file)
        
        return model


    def train_naive_bayes(self, preprocessor=None, file_path=None):
        """
        Trains a Multinomial Naive Bayes model using the preprocessed and vectorized data.
        
        Args:
            preprocessor: Optional DataPreprocessor object with preprocessed data
            file_path: Path to the processed data file if preprocessor is None
            
        Returns:
            A dictionary containing the trained model, vectorizer and metrics
        """
        print("Starting Naive Bayes model training...")
        
        # Prepare data
        if preprocessor is None and file_path is not None:
            preprocessor = DataPreprocessor(file_path)
            preprocessor.clean_data()
            preprocessor.split_data()
        
        if preprocessor is None:
            default_path = os.path.join(PROCESSED_DATA_PATH, "processed_dataset.csv")
            preprocessor = DataPreprocessor(default_path)
            preprocessor.clean_data()
            preprocessor.split_data()
        
        # Vectorize the text data using the preprocessor's vectorizer
        (X_train_vec, X_test_vec), vectorizer = preprocessor.vectorize_text()
        y_train = preprocessor.y_train
        y_test = preprocessor.y_test
        
        # Calculate class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        # Create class weight dictionary
        weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Class weights for balancing: {weight_dict}")
        
        # Train the model with hyperparameter tuning
        print("Training Multinomial Naive Bayes model...")
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
            'fit_prior': [True, False]
        }
        
        nb_model = MultinomialNB()
        grid_search = GridSearchCV(nb_model, param_grid, cv=5, scoring='f1_weighted')
        grid_search.fit(X_train_vec, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Best hyperparameters: {grid_search.best_params_}")
        
        # Evaluate the model
        y_pred = best_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Check class distribution in predictions
        print("Prediction class distribution:")
        unique, counts = np.unique(y_pred, return_counts=True)
        for class_idx, count in zip(unique, counts):
            print(f"Class {class_idx}: {count} predictions")
        
        # Save model components
        model_data = {
            'model': best_model,
            'vectorizer': vectorizer,
            'label_encoder': preprocessor.label_encoder,
            'metrics': {
                'accuracy': accuracy,
                'best_params': grid_search.best_params_
            }
        }
        
        # Save the trained model
        save_path = os.path.join(MODEL_DIR, 'naive_bayes_model.pkl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Naive Bayes model saved at: {save_path}")
        
        # Store in the instance for later use
        self.naive_bayes_model = model_data
        
        return model_data

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
