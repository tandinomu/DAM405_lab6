"""
Task 6 - Load the registered model and predict.

Loads the latest registered version of 'wine-classifier' from the
Model Registry (not from a run) and predicts on 5 test samples.
"""
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from common import MODEL_NAME, setup

setup()

# same split as training, so Xte is unseen data
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
print("Loaded model prediction:", model.predict(Xte[:5]))
print("True labels:            ", yte[:5])
