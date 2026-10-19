"""
Shared settings for Practical 6.

Every script uses the same tracking store so the UI, the training runs and
the registry all see the same data. A local SQLite file (mlflow.db) is used
because the Model Registry needs a database-backed store.
"""
import os

import mlflow

BASE = os.path.dirname(os.path.abspath(__file__))
TRACKING_URI = "sqlite:///" + os.path.join(BASE, "mlflow.db")
EXPERIMENT = "wine-classifier"
MODEL_NAME = "wine-classifier"


def setup():
    """Point MLflow at the local store and select the experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
