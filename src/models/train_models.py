import mlflow
import pandas as pd

from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import *

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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
    def svm(C=1.0):
        experiment_id = ModelTrainer._setup_experiment()
        with mlflow.start_run(experiment_id=experiment_id):
            raise NotImplementedError("Training logic not implemented")

    @staticmethod
    def kernel_svm(gamma='scale'):
        experiment_id = ModelTrainer._setup_experiment()
        with mlflow.start_run(experiment_id=experiment_id):
            raise NotImplementedError("Training logic not implemented")

    @staticmethod
    def crf():
        experiment_id = ModelTrainer._setup_experiment()
        with mlflow.start_run(experiment_id=experiment_id):
            raise NotImplementedError("Training logic not implemented")