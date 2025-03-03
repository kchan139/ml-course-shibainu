# src/models/train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from pgmpy.models import BayesianNetwork  # or BayesianModel (deprecated, use BayesianNetwork)
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from hmmlearn import hmm
from datetime import datetime
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from ..features.build_features import FeatureExtractor
from src.data.preprocess import DataPreprocessor
from src.config import *

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

    def train_neural_network(self, preprocessor=None, file_path=None, max_words=10000, 
                             embedding_dim=100, max_len=100, epochs=10, batch_size=64):
        """
        Trains a neural network (CNN-BiLSTM) model for sentiment classification.
        
        Args:
            preprocessor: Optional DataPreprocessor object with preprocessed data
            file_path: Path to the processed data file if preprocessor is None
            max_words: Maximum number of words for tokenization
            embedding_dim: Dimension for the embedding layer
            max_len: Maximum length of sequences
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Trained neural network model
            
        Saves:
            The trained model and tokenizer in pickle format
        """
        print("Starting neural network training...")
        
        # Prepare data
        if preprocessor is None and file_path is not None:
            # Create and initialize the preprocessor
            preprocessor = DataPreprocessor(file_path)
            preprocessor.clean_data()
            preprocessor.split_data()
        
        if preprocessor is None:
            # Use default processed data path if no preprocessor or file provided
            default_path = os.path.join(PROCESSED_DATA_PATH, "processed_dataset.csv")
            preprocessor = DataPreprocessor(default_path)
            preprocessor.clean_data()
            preprocessor.split_data()
        
        # Access the training and test data
        X_train = preprocessor.X_train
        X_test = preprocessor.X_test
        y_train = preprocessor.y_train
        y_test = preprocessor.y_test
        
        # Convert to categorical format for neural network
        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        # Tokenize text data
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences to ensure uniform length
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
        
        # Define the CNN-BiLSTM model architecture
        model = Sequential()
        
        # Embedding layer
        model.add(Embedding(max_words, embedding_dim, input_length=max_len))
        
        # CNN layers
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(64, 5, activation='relu'))
        
        # BiLSTM layer
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        
        # Global pooling and dense layers
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train_pad, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_pad, y_test_cat),
            callbacks=[early_stopping]
        )
        
        # Evaluate the model
        y_pred_prob = model.predict(X_test_pad)
        y_pred = np.argmax(y_pred_prob, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save the model and tokenizer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"neural_network_{timestamp}.pkl"
        model_path = EXPERIMENT_DIR + 'trained' + model_filename
        
        model_data = {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': preprocessor.label_encoder,
            'max_len': max_len,
            'metrics': {
                'accuracy': accuracy
            }
        }
        
        with open(model_path, 'wb') as file:
            pickle.dump(model_data, file)
        
        return model


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
        import os
        import numpy as np
        from hmmlearn import hmm
        import pickle
        from src.config import MODEL_DIR
        
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