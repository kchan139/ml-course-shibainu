import mlflow
import numpy as np

class ModelPredictor:
    @staticmethod
    def predict(run_id: str, X: np.ndarray):
        """Make predictions using a trained model"""
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        return model.predict(X)