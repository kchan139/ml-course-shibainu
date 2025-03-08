# src/models/predict_model.py
import os
import re
import string
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pgmpy.inference import VariableElimination
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

from src.config import *
from src.data.preprocess import DataPreprocessor

class ModelPredictor:
    """
    This class handles making predictions using the trained models.
    """

    def predict_decision_tree(self, X_test, model_trainer):
        """
        Makes predictions using the trained Decision Tree model stored in the provided ModelTrainer instance.
        
        Args:
            X_test: Feature matrix (e.g., array or sparse matrix) for the test data.
            model_trainer: An instance of ModelTrainer that has a trained decision tree model (e.g., 
                           accessible via model_trainer.decision_tree_model).
        
        Returns:
            Array-like predicted labels for the test data.
        """
        # Use the trained decision tree model from the ModelTrainer instance to predict labels.
        predictions = model_trainer.decision_tree_model.predict(X_test)
        return predictions
    

    def predict_neural_network(self, headlines, model_path=None, custom_threshold=0.35):
        """
        Makes predictions using the trained Neural Network model with custom thresholding
        to better handle class imbalance.
        
        Args:
            headlines: A string or list of strings containing news headlines to predict
            model_path: Optional path to a specific model file. If None, uses the most recent model.
            custom_threshold: Confidence threshold for class prediction (default: 0.35)
            
        Returns:
            A list of dictionaries containing the original headline, predicted sentiment, 
            and confidence scores.
        """
        # Convert single headline to list for consistent processing
        if isinstance(headlines, str):
            headlines = [headlines]
            
        # Find the most recent model if no path is provided
        if model_path is None:
            model_dir = Path(MODEL_DIR)
            models = list(model_dir.glob('*rnn_*.pkl'))
            if not models:
                model_dir = Path(EXPERIMENT_DIR)
                models = list(model_dir.glob('*rnn_*.pkl'))
            if not models:
                print(f"No models found in {model_dir}")
                return None
                
            # Sort models by creation time and get the most recent
            model_path = str(sorted(models, key=os.path.getmtime)[-1])
            print(f"Using most recent model: {model_path}")
        
        # Load the model
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Extract model components
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            label_encoder = model_data['label_encoder']
            max_len = model_data.get('max_len', 100)  # Default to 100 if not specified
            
            # Use saved threshold if available, otherwise use the provided one
            threshold = model_data.get('custom_threshold', custom_threshold)
            
            # Process the headlines
            sequences = tokenizer.texts_to_sequences(headlines)
            padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
            
            # Generate raw prediction probabilities
            prediction_probs = model.predict(padded_sequences)
            
            # Process results with custom threshold logic
            results = []
            class_distribution = {0: 0, 1: 0, 2: 0}  # Track prediction distribution
            
            for i, headline in enumerate(headlines):
                probs = prediction_probs[i]
                
                # Apply custom threshold logic to combat class imbalance
                # If no class exceeds threshold, pick highest probability
                if not any(p > threshold for p in probs):
                    predicted_class_idx = np.argmax(probs)
                # Otherwise use custom logic to favor minority classes
                elif probs[0] > threshold*0.8:  # Lower threshold for negative class
                    predicted_class_idx = 0
                elif probs[2] > threshold*0.8:  # Lower threshold for positive class
                    predicted_class_idx = 2
                else:
                    predicted_class_idx = np.argmax(probs)
                
                # Update class distribution counter
                if predicted_class_idx in class_distribution:
                    class_distribution[predicted_class_idx] += 1
                    
                predicted_sentiment = label_encoder.inverse_transform([predicted_class_idx])[0]
                
                # Create result dictionary
                result = {
                    'headline': headline,
                    'sentiment': predicted_sentiment,
                    'confidence': float(probs[predicted_class_idx]),
                    'probabilities': {
                        label_encoder.inverse_transform([j])[0]: float(prob) 
                        for j, prob in enumerate(probs)
                    }
                }
                results.append(result)
            
            # Print class distribution for monitoring
            print("Prediction class distribution:")
            for class_idx, count in class_distribution.items():
                sentiment = label_encoder.inverse_transform([class_idx])[0]
                print(f"{sentiment} (class {class_idx}): {count} predictions")
            
            return results
            
        except Exception as e:
            print(f"Error predicting with neural network: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        

    def predict_naive_bayes(self, headlines, model_trainer=None, model_path=None):
        """
        Makes predictions using the trained Naive Bayes model.
        
        Args:
            headlines: A string or list of strings containing news headlines to predict
            model_trainer: Optional ModelTrainer instance with trained naive_bayes_model
            model_path: Optional path to a specific model file
            
        Returns:
            A list of dictionaries containing headline, predicted sentiment, and probabilities
        """
        # Convert single headline to list for consistent processing
        if isinstance(headlines, str):
            headlines = [headlines]
        
        # Get model data - either from trainer, specific path, or default path
        model_data = None
        
        if model_trainer is not None and hasattr(model_trainer, 'naive_bayes_model'):
            model_data = model_trainer.naive_bayes_model
            print("Using Naive Bayes model from model_trainer")
        elif model_path is not None:
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                print(f"Loaded Naive Bayes model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {str(e)}")
        else:
            # Try default path
            default_path = os.path.join(MODEL_DIR, 'naive_bayes_model.pkl')
            try:
                with open(default_path, 'rb') as f:
                    model_data = pickle.load(f)
                print(f"Loaded Naive Bayes model from default path: {default_path}")
            except FileNotFoundError:
                print(f"No model found at {default_path}. Training a new model...")
                # Train a new model if none exists
                from src.models.train_model import ModelTrainer
                from src.config import PROCESSED_DATA_PATH
                
                default_data_path = os.path.join(PROCESSED_DATA_PATH, "processed_dataset.csv")
                preprocessor = DataPreprocessor(default_data_path)
                preprocessor.clean_data()
                preprocessor.split_data()
                
                trainer = ModelTrainer()
                model_data = trainer.train_naive_bayes(preprocessor=preprocessor)
        
        # Now make predictions
        if model_data is None:
            print("Failed to load or train a Naive Bayes model.")
            return None
        
        # Extract components
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        label_encoder = model_data['label_encoder']
        
        # Clean headlines using the same preprocessing steps
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
        
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'\d+', '', text)
            text = text.translate(str.maketrans("", "", string.punctuation))
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
            return " ".join(tokens)
        
        cleaned_headlines = [clean_text(headline) for headline in headlines]
        
        # Transform headlines to TF-IDF vectors using the saved vectorizer
        X_headlines = vectorizer.transform(cleaned_headlines)
        
        # Get predictions and probabilities
        y_pred = model.predict(X_headlines)
        y_pred_proba = model.predict_proba(X_headlines)
        
        # Build result dictionaries
        results = []
        for i, headline in enumerate(headlines):
            pred_class_idx = y_pred[i]
            sentiment = label_encoder.inverse_transform([pred_class_idx])[0]
            
            # Create probability dictionary for each class
            prob_dict = {
                label_encoder.inverse_transform([j])[0]: float(prob)
                for j, prob in enumerate(y_pred_proba[i])
            }
            
            result = {
                'headline': headline,
                'sentiment': sentiment,
                'confidence': float(y_pred_proba[i][pred_class_idx]),
                'probabilities': prob_dict
            }
            results.append(result)
        
        return results


    def predict_bayesian_network(self, test_text_data, model_trainer=None, model_path=None):
        """
        Args:
            test_text_data (iterable): Cleaned text strings to predict on.
            model_trainer (object, optional): Instance of ModelTrainer with
                - trained_model
                - count_vectorizer (cv)
                - selector
                - selected_features
              Defaults to None.
            model_path (str, optional): Path to the file containing all artifacts (model, vectorizer, selector, features).
              Defaults to None.
        
        Returns:
            list: Predicted labels for each text in 'test_text_data'.
        """

        # 1) Determine how to load artifacts
        if model_trainer is not None:
            # Use artifacts in memory from model_trainer
            trained_model = model_trainer.trained_model
            cv = model_trainer.cv
            selector = model_trainer.selector
            selected_features = model_trainer.selected_features

        else:
            # If model_trainer is None, check model_path
            if model_path is None:
                model_path = os.path.join(MODEL_DIR, 'bayesian_network_model.pkl')  

            # Load artifacts from file
            with open(model_path, 'rb') as f:
                artifacts = pickle.load(f)
            trained_model = artifacts['trained_model']
            cv = artifacts['count_vectorizer']
            selector = artifacts['selector']
            selected_features = artifacts['selected_features']

        # 2) Transform test_text_data into features
        # Transform the test text using the fitted CountVectorizer
        X_counts_test = cv.transform(test_text_data)

        # Apply the same feature selection
        X_selected_test = selector.transform(X_counts_test)

        # Convert to binary presence/absence (as in your original code)
        X_features_test = (X_selected_test > 0).astype(int)

        # Create a DataFrame with the selected feature names
        df_test = pd.DataFrame(X_features_test.toarray(), columns=selected_features)

        # 3) Perform inference using the trained model
        infer = VariableElimination(trained_model)
        predictions = []

        for _, row in df_test.iterrows():
            # Build the evidence dictionary from the row (convert values to int)
            evidence = {col: int(row[col]) for col in df_test.columns}

            # Query the network for the 'label' variable
            query_result = infer.query(variables=["label"], evidence=evidence)

            # Pick the label with the highest probability
            predicted_label = query_result.values.argmax()
            predictions.append(predicted_label)

        return predictions
    
    def predict_bayesian_network_W2V(self, test_text_data, model_trainer=None, model_path=None):
        """
        Predicts labels using the trained Bayesian Network and word2vec-based sentence embeddings.
        Args:
            test_text_data (list or Series of str): Cleaned text strings to predict on.
            model_trainer (object, optional): Instance of ModelTrainer containing:
                - trained_model (BayesianNetwork)
                - w2v_model (Word2Vec)
                - discretizer (KBinsDiscretizer)
              Defaults to None.
            model_path (str, optional): Path to the file containing all artifacts. Defaults to None.

        Returns:
            list: Predicted labels for each text in 'test_text_data'.
        """

        # ----------------------------
        # 1) Determine how to load artifacts
        # ----------------------------
        if model_trainer is not None:
            # Use artifacts directly from model_trainer
            trained_model = model_trainer.trained_model
            w2v_model = model_trainer.w2v_model
            discretizer = model_trainer.discretizer
        else:
            # If model_trainer is None, check model_path
            if model_path is None:
                # Use a default path
                model_path = os.path.join(MODEL_DIR, 'bayesian_network_W2V.pkl')

            # Load artifacts from the saved file
            with open(model_path, 'rb') as f:
                artifacts = pickle.load(f)

            trained_model = artifacts['trained_model']
            w2v_model = artifacts['word2vec_model']
            discretizer = artifacts['discretizer']
            # If you saved feature_names, you can load them here as well:
            # feature_names = artifacts['feature_names']

        # Helper function to compute the average word embeddings
        def sentence_embedding(sentence, model):
            words = sentence.split()
            vectors = [model.wv[word] for word in words if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

        # Convert test text into embeddings
        X_w2v_test = np.array([sentence_embedding(text, w2v_model) for text in test_text_data])

        # Discretize using the trained discretizer
        X_discrete_test = discretizer.transform(X_w2v_test)

        # Clip values to avoid unseen bin numbers
        X_discrete_test = np.clip(X_discrete_test, 0, discretizer.n_bins_[0] - 1)

        # Create a DataFrame for the discretized features
        feature_names = [f"feat_{i}" for i in range(X_discrete_test.shape[1])]
        df_test = pd.DataFrame(X_discrete_test, columns=feature_names)

        # Ensure only features that exist in the trained Bayesian Network are used
        model_nodes = list(trained_model.nodes())
        valid_columns = list(set(model_nodes) & set(df_test.columns))
        df_test = df_test[valid_columns]

        # Perform inference
        infer = VariableElimination(trained_model)
        predictions = []
        for _, row in df_test.iterrows():
            evidence = {col: int(row[col]) for col in df_test.columns}
            query_result = infer.query(variables=["label"], evidence=evidence)
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
