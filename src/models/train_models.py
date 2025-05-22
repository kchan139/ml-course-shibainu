import mlflow
import pandas as pd
import numpy as np
import os

from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModelContext

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import *

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Transform text data into feature dictionaries for CRF"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        
    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self
        
    def transform(self, X):
        # Transform text to TF-IDF features
        tfidf_features = self.vectorizer.transform(X)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Convert to feature dictionaries for CRF
        sequences = []
        for sample in tfidf_features:
            # Get non-zero feature indices and values
            indices = sample.nonzero()[1]
            values = sample.data
            # Create feature dictionary
            features = {
                str(feature_names[idx]): str(round(val, 4))
                for idx, val in zip(indices, values)
                if val > 0
            }
            sequences.append([features])  # Single token sequence
        return sequences

class LabelTransformer(BaseEstimator, TransformerMixin):
    """Transform labels to sequence format for CRF"""
    def fit(self, y):
        return self
        
    def transform(self, y):
        # Convert integer labels to sequences of strings for CRF
        return [[str(label)] for label in y]
        
    def inverse_transform(self, y):
        # Convert sequence predictions back to integers
        return np.array([int(seq[0]) for seq in y])

class ModelTrainer:
    @staticmethod
    def _setup_experiment(name="sentiment_classification"):
        """Setup or get existing active experiment, or restore if deleted."""
        client = MlflowClient()
        experiment = client.get_experiment_by_name(name)

        if experiment is None:
            # Experiment does not exist, create it
            print(f"Creating new MLflow experiment '{name}'...")
            return client.create_experiment(name)
        elif experiment.lifecycle_stage == 'active':
            # Experiment exists and is active
            print(f"Using active MLflow experiment '{name}' with ID {experiment.experiment_id}")
            return experiment.experiment_id
        elif experiment.lifecycle_stage == 'deleted':
            # Experiment exists but is deleted, attempt to restore it
            print(f"MLflow experiment '{name}' (ID: {experiment.experiment_id}) is deleted. Attempting to restore...")
            try:
                client.restore_experiment(experiment.experiment_id)
                print(f"Successfully restored MLflow experiment '{name}' with ID {experiment.experiment_id}")
                return experiment.experiment_id
            except Exception as e:
                print(f"Failed to restore MLflow experiment '{name}' (ID: {experiment.experiment_id}). Error: {e}")
                print("To resolve, consider manually deleting the experiment permanently from MLflow UI or the '.trash' directory, then re-running.")
                # Fallback: if restoration fails, create a new experiment with a unique name
                new_name = f"{name}_recreated_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                print(f"Creating a new MLflow experiment with a unique name: '{new_name}'")
                return client.create_experiment(new_name)
        else:
            # Fallback for unexpected lifecycle stages
            print(f"MLflow experiment '{name}' is in an unexpected lifecycle stage: {experiment.lifecycle_stage}. Creating a new experiment.")
            new_name = f"{name}_recreated_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
            return client.create_experiment(new_name)

    @staticmethod
    def logistic_regression(C=1.0, penalty='l2', solver='liblinear',
                            max_iter=100, test_size=0.2, random_state=42):
        """Train a logistic regression model with text data"""
        experiment_id = ModelTrainer._setup_experiment()

        with mlflow.start_run(experiment_id=experiment_id):
            # Load and prepare data
            df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'Reviews_preprocessed.csv'))
            df['target'] = (df['Score'] >= 4).astype(int)

            X_text = df['Text_Normalized'].fillna('').values
            y = df['target'].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Create model pipeline
            model = Pipeline([
                ('vectorizer', TfidfVectorizer()),
                ('classifier', LogisticRegression(C=C, penalty=penalty, solver=solver,
                                                max_iter=max_iter))
            ])

            # Training
            model.fit(X_train, y_train)
            mlflow.log_param("C", C)
            mlflow.log_param("penalty", penalty)
            mlflow.log_param("solver", solver)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)

            # Predictions on train set and metrics
            train_preds = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_preds)
            train_precision = precision_score(y_train, train_preds, average='weighted', zero_division=0)
            train_recall = recall_score(y_train, train_preds, average='weighted', zero_division=0)
            train_f1 = f1_score(y_train, train_preds, average='weighted', zero_division=0)

            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("train_f1", train_f1)

            # Predictions on test set and metrics
            test_preds = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_preds)
            test_precision = precision_score(y_test, test_preds, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, test_preds, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)

            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1", test_f1)

            # Create signature and input example using a sample of the training data
            signature_input_sample = X_train[:5] if len(X_train) >= 5 else X_train
            signature_output_sample = model.predict(signature_input_sample)

            signature = infer_signature(signature_input_sample, signature_output_sample)
            input_example = X_train[:3] if len(X_train) >=3 else X_train

            # Log model with signature and example
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=input_example
            )

            return mlflow.active_run().info.run_id

    @staticmethod
    def gradient_boosting(n_estimators=100):
        experiment_id = ModelTrainer._setup_experiment()
        with mlflow.start_run(experiment_id=experiment_id):
            raise NotImplementedError("Training logic not implemented")

    @staticmethod
    def random_forest(n_estimators=100):
        experiment_id = ModelTrainer._setup_experiment()
        with mlflow.start_run(experiment_id=experiment_id):
            raise NotImplementedError("Training logic not implemented")

    @staticmethod
    def svm(C=1.0, penalty='l2', max_iter=100, test_size=0.2, random_state=42):
        """Train a linear SVM model with text data"""
        experiment_id = ModelTrainer._setup_experiment()

        with mlflow.start_run(experiment_id=experiment_id):
            # Load and prepare data
            df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'Reviews_preprocessed.csv'))
            df['target'] = (df['Score'] >= 4).astype(int)

            X_text = df['Text_Normalized'].fillna('').values
            y = df['target'].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Create model pipeline
            model = Pipeline([
                ('vectorizer', TfidfVectorizer()),
                ('classifier', LinearSVC(C=C, penalty=penalty, max_iter=max_iter))
            ])

            # Training
            model.fit(X_train, y_train)
            mlflow.log_param("C", C)
            mlflow.log_param("penalty", penalty)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)

            # Predictions on train set and metrics
            train_preds = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_preds)
            train_precision = precision_score(y_train, train_preds, average='weighted', zero_division=0)
            train_recall = recall_score(y_train, train_preds, average='weighted', zero_division=0)
            train_f1 = f1_score(y_train, train_preds, average='weighted', zero_division=0)

            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("train_f1", train_f1)

            # Predictions on test set and metrics
            test_preds = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_preds)
            test_precision = precision_score(y_test, test_preds, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, test_preds, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)

            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1", test_f1)

            # Create signature and input example
            signature_input_sample = X_train[:5] if len(X_train) >= 5 else X_train
            signature_output_sample = model.predict(signature_input_sample)
            signature = infer_signature(signature_input_sample, signature_output_sample)
            input_example = X_train[:3] if len(X_train) >= 3 else X_train

            # Log model with signature and example
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=input_example
            )

            return mlflow.active_run().info.run_id

    @staticmethod
    def kernel_svm(C=1.0, kernel='rbf', gamma='scale', max_iter=-1, test_size=0.2, random_state=42):
        """Train a kernel SVM model with text data"""
        experiment_id = ModelTrainer._setup_experiment()

        with mlflow.start_run(experiment_id=experiment_id):
            # Load and prepare data
            df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'Reviews_preprocessed.csv'))
            df['target'] = (df['Score'] >= 4).astype(int)

            X_text = df['Text_Normalized'].fillna('').values
            y = df['target'].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Create model pipeline
            model = Pipeline([
                ('vectorizer', TfidfVectorizer()),
                ('classifier', SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iter))
            ])

            # Training
            model.fit(X_train, y_train)
            mlflow.log_param("C", C)
            mlflow.log_param("kernel", kernel)
            mlflow.log_param("gamma", gamma)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)

            # Predictions on train set and metrics
            train_preds = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_preds)
            train_precision = precision_score(y_train, train_preds, average='weighted', zero_division=0)
            train_recall = recall_score(y_train, train_preds, average='weighted', zero_division=0)
            train_f1 = f1_score(y_train, train_preds, average='weighted', zero_division=0)

            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("train_f1", train_f1)

            # Predictions on test set and metrics
            test_preds = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_preds)
            test_precision = precision_score(y_test, test_preds, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, test_preds, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)

            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1", test_f1)

            # Create signature and input example
            signature_input_sample = X_train[:5] if len(X_train) >= 5 else X_train
            signature_output_sample = model.predict(signature_input_sample)
            signature = infer_signature(signature_input_sample, signature_output_sample)
            input_example = X_train[:3] if len(X_train) >= 3 else X_train

            # Log model with signature and example
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=input_example
            )

            return mlflow.active_run().info.run_id

    @staticmethod
    def crf(c1=0.1, c2=0.1, max_iterations=100, test_size=0.2, random_state=42):
        """Train a CRF model with text data and log it to MLflow."""
        experiment_id = ModelTrainer._setup_experiment()

        with mlflow.start_run(experiment_id=experiment_id):
            # ─────────────────── data prep ─────────────────── #
            df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH,
                                        "Reviews_preprocessed.csv"))
            df["target"] = (df["Score"] >= 4).astype(int)

            X_text = df["Text_Normalized"].fillna("").values
            y       = df["target"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y, test_size=test_size, random_state=random_state,
                stratify=y
            )

            feature_tf   = FeatureTransformer()
            label_tf     = LabelTransformer()

            X_train_seq  = feature_tf.fit_transform(X_train)
            X_test_seq   = feature_tf.transform(X_test)
            y_train_seq  = label_tf.transform(y_train)
            y_test_seq   = label_tf.transform(y_test)

            crf_model = CRF(
                algorithm="lbfgs",
                c1=c1,
                c2=c2,
                max_iterations=max_iterations,
                all_possible_transitions=True,
                verbose=False,
            )
            crf_model.fit(X_train_seq, y_train_seq)

            # ─────────────────── metrics ─────────────────── #
            train_preds = label_tf.inverse_transform(crf_model.predict(X_train_seq))
            test_preds  = label_tf.inverse_transform(crf_model.predict(X_test_seq))

            mlflow.log_params({
                "c1": c1, "c2": c2,
                "max_iterations": max_iterations,
                "test_size": test_size,
                "random_state": random_state
            })
            mlflow.log_metrics({
                "train_accuracy":  accuracy_score(y_train, train_preds),
                "train_precision": precision_score(y_train, train_preds,
                                                average="weighted",
                                                zero_division=0),
                "train_recall":    recall_score(y_train, train_preds,
                                                average="weighted",
                                                zero_division=0),
                "train_f1":        f1_score(y_train, train_preds,
                                            average="weighted",
                                            zero_division=0),
                "test_accuracy":   accuracy_score(y_test,  test_preds),
                "test_precision":  precision_score(y_test, test_preds,
                                                average="weighted",
                                                zero_division=0),
                "test_recall":     recall_score(y_test,  test_preds,
                                                average="weighted",
                                                zero_division=0),
                "test_f1":         f1_score(y_test,  test_preds,
                                            average="weighted",
                                            zero_division=0),
            })

            # ──────── wrap model in a proper PythonModel ──────── #
            class CRFPyfuncModel(mlflow.pyfunc.PythonModel):
                def __init__(self, feat_tf, crf, lbl_tf):
                    self.feat_tf = feat_tf
                    self.crf     = crf
                    self.lbl_tf  = lbl_tf

                def predict(
                    self, 
                    context: PythonModelContext, 
                    model_input: pd.Series  # or pd.DataFrame, np.ndarray, List[str], etc.
                ) -> pd.Series:              # or np.ndarray, List[int], etc.
                    """
                    model_input: raw texts in a pandas Series
                    returns: pandas Series of integer predictions
                    """
                    X_seq = self.feat_tf.transform(model_input.tolist())
                    y_seq = self.crf.predict(X_seq)
                    return pd.Series(self.lbl_tf.inverse_transform(y_seq))

            pyfunc_model = CRFPyfuncModel(feature_tf, crf_model, label_tf)

            # create model signature
            signature = infer_signature(X_train[:5], pyfunc_model.predict(None, X_train[:5]))
            input_example = X_train[:3]

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pyfunc_model,
                signature=signature,
                input_example=input_example
            )

            return mlflow.active_run().info.run_id