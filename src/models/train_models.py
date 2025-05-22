import mlflow
import pandas as pd

from mlflow.models.signature import infer_signature

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.data.preprocess import DataLoader
from src.config import *

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class ModelTrainer:
    @staticmethod
    def _setup_experiment(name="sentiment_classification"):
        """Setup or get existing experiment"""
        try:
            experiment_id = mlflow.create_experiment(name)
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(name).experiment_id
        return experiment_id

    @staticmethod
    def logistic_regression(C=1.0, test_size=0.2, random_state=42): # Added test_size and random_state
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
                ('classifier', LogisticRegression(C=C, solver='liblinear')) # Added solver for potentially large datasets
            ])

            # Training
            model.fit(X_train, y_train)
            mlflow.log_param("C", C)
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
            # Using X_train for infer_signature as the model was trained on it.
            # Predictions for signature can be from a sample of X_train
            signature_input_sample = X_train[:5] if len(X_train) >= 5 else X_train
            signature_output_sample = model.predict(signature_input_sample)

            signature = infer_signature(signature_input_sample, signature_output_sample)
            input_example = X_train[:3] if len(X_train) >=3 else X_train # Ensure there are enough samples

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
            return ModelTrainer._base_train("gbt", {"n_estimators": n_estimators})

    @staticmethod
    def random_forest(n_estimators=100):
        experiment_id = ModelTrainer._setup_experiment()
        with mlflow.start_run(experiment_id=experiment_id):
            raise NotImplementedError("Training logic not implemented")
            return ModelTrainer._base_train("rf", {"n_estimators": n_estimators})

    @staticmethod
    def svm(C=1.0):
        experiment_id = ModelTrainer._setup_experiment()
        with mlflow.start_run(experiment_id=experiment_id):
            raise NotImplementedError("Training logic not implemented")
            return ModelTrainer._base_train("svm", {"C": C})

    @staticmethod
    def kernel_svm(gamma='scale'):
        experiment_id = ModelTrainer._setup_experiment()
        with mlflow.start_run(experiment_id=experiment_id):
            raise NotImplementedError("Training logic not implemented")
            return ModelTrainer._base_train("kernel_svm", {"gamma": gamma})

    @staticmethod
    def crf():
        experiment_id = ModelTrainer._setup_experiment()
        with mlflow.start_run(experiment_id=experiment_id):
            raise NotImplementedError("Training logic not implemented")
            sequences = [["Sample", "sequence"]]
            DataLoader.crf_preprocess(sequences)
            return ModelTrainer._base_train("crf", {})