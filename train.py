"""
Task 2 - Track a training run, and Task 3 - Run several experiments.

Each run logs:
  * parameters : n_estimators, max_depth
  * metric     : accuracy on the test set
  * artifact   : the trained model

Run:  python train.py      -> 9 runs (3 x 3 hyperparameter grid)
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

from common import setup

# ---------- data ----------
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

setup()  # tracking store + experiment 'wine-classifier'


def run(n_estimators, max_depth):
    """Train one model inside an MLflow run and log everything about it."""
    with mlflow.start_run(run_name=f"rf_n{n_estimators}_d{max_depth}"):
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     random_state=42).fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        # Extra metric: test accuracy is 1.0 for most settings on this small
        # dataset, so 5-fold cross-validation accuracy on the training data
        # is logged as a tie-breaker that actually separates the runs.
        cv_acc = cross_val_score(
            RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=42), Xtr, ytr, cv=5).mean()

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("cv_accuracy", cv_acc)
        # input_example lets MLflow record the model's input signature.
        # cloudpickle format: newer MLflow defaults to 'skops', which refuses
        # to save tree models (RandomForest) without extra trust settings.
        mlflow.sklearn.log_model(
            clf, name="model", input_example=Xtr[:2],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

        print(f"n={n_estimators} depth={max_depth} acc={acc:.4f} cv_acc={cv_acc:.4f}")


if __name__ == "__main__":
    # Task 3: vary the hyperparameters to get multiple comparable runs
    for n in (50, 100, 200):
        for d in (3, 5, None):
            run(n, d)
    print("\nDone. Start the UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
