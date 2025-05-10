import mlflow
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from src.config import MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_PATH
from src.data.preprocess import DataLoader

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class ModelTrainer:
    @staticmethod
    def _base_train(model_type: str, params: dict):
        """Shared training logic for all models"""

        raise NotImplementedError("Training logic not implemented")

        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model.fit(X, y)
        
        # Evaluate
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        mlflow.log_metric("train_accuracy", acc)
        
        # Save model
        mlflow.sklearn.log_model(model, "model")
        return mlflow.active_run().info.run_id

    @staticmethod
    def logistic_regression(C=1.0):
        with mlflow.start_run(experiment_id=0):
            raise NotImplementedError("Training logic not implemented")
            return ModelTrainer._base_train("logistic", {"C": C})

    @staticmethod
    def gradient_boosting(n_estimators=100):
        with mlflow.start_run(experiment_id=0):
            raise NotImplementedError("Training logic not implemented")
            return ModelTrainer._base_train("gbt", {"n_estimators": n_estimators})

    @staticmethod
    def random_forest(n_estimators=100):
        with mlflow.start_run(experiment_id=0):
            raise NotImplementedError("Training logic not implemented")
            return ModelTrainer._base_train("rf", {"n_estimators": n_estimators})

    @staticmethod
    def svm(C=1.0):
        with mlflow.start_run(experiment_id=0):
            raise NotImplementedError("Training logic not implemented")
            return ModelTrainer._base_train("svm", {"C": C})

    @staticmethod
    def kernel_svm(gamma='scale'):
        with mlflow.start_run(experiment_id=0):
            raise NotImplementedError("Training logic not implemented")
            return ModelTrainer._base_train("kernel_svm", {"gamma": gamma})

    @staticmethod
    def crf():
        with mlflow.start_run(experiment_id=0):
            raise NotImplementedError("Training logic not implemented")
            sequences = [["Sample", "sequence"]]
            DataLoader.crf_preprocess(sequences)
            return ModelTrainer._base_train("crf", {})