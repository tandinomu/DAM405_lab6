"""
Task 5 - Register the best model.

Finds the run with the highest accuracy automatically (instead of copying
the RUN_ID by hand) and registers its model as 'wine-classifier'.
Running it again creates a new version (version 2, 3, ...).
"""
import mlflow

from common import EXPERIMENT, MODEL_NAME, setup

setup()

# Sort all runs by test accuracy, then by cross-validation accuracy (tie-breaker)
runs = mlflow.search_runs(experiment_names=[EXPERIMENT],
                          order_by=["metrics.accuracy DESC", "metrics.cv_accuracy DESC"])
best = runs.iloc[0]
run_id = best["run_id"]
print(f"Best run: {run_id}")
print(f"  n_estimators={best['params.n_estimators']} "
      f"max_depth={best['params.max_depth']} "
      f"accuracy={best['metrics.accuracy']:.4f} "
      f"cv_accuracy={best['metrics.cv_accuracy']:.4f}")

# Same call as the handout, with the best RUN_ID filled in
result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
print("Registered version:", result.version)
