# Practical 6 - Managing the ML Lifecycle with MLflow

Trains a Random Forest on the Wine dataset, tracks 9 runs with MLflow,
registers the best model in the Model Registry, and loads it back to predict.

## Files

| File | Task |
|---|---|
| `common.py` | Shared tracking store (`mlflow.db`) and experiment name |
| `train.py` | Tasks 2–3: logs params, accuracy and model for 9 runs |
| `register_model.py` | Task 5: finds the best run and registers it as `wine-classifier` |
| `load_model.py` | Task 6: loads `models:/wine-classifier/latest` and predicts |

## Setup (macOS )

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mlflow --version
```

## Run

```bash
python train.py                     # 9 runs
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open http://localhost:5000 and sort the runs by accuracy (Task 4).
Leave the UI running, open a second terminal (activate venv again):

```bash
python register_model.py            # registers the best run
python load_model.py                # loads it from the registry and predicts
```

If port 5000 is busy (macOS AirPlay Receiver uses it), use `--port 5001`.

## Optional: serve over REST

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow models serve -m "models:/wine-classifier/latest" -p 5002 --env-manager local
```
