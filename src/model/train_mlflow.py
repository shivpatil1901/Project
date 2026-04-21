"""
Train a single model with MLflow tracking.
Usage:
  python src/model/train_mlflow.py --model-name random_forest
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


def load_params(params_file="params.yaml"):
    with open(params_file, "r") as f:
        return yaml.safe_load(f)


def sanitize_param_value(value):
    """Convert non-scalar values to strings for MLflow param logging."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value)


def build_run_name(model_name, model_cfg):
    """Build a stable MLflow run name from model name and hyperparameters."""
    parts = [f"train_{model_name}"]
    for key in sorted(model_cfg.keys()):
        value = model_cfg[key]
        parts.append(f"{key}={sanitize_param_value(value)}")
    return "_".join(parts)


def get_model(model_name, params):
    cfg = dict(params["models"].get(model_name, {}))

    if model_name == "logistic_regression":
        cfg.setdefault("max_iter", 1000)
        return LogisticRegression(**cfg), cfg

    if model_name == "random_forest":
        cfg.setdefault("n_jobs", -1)
        return RandomForestClassifier(**cfg), cfg

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(**cfg), cfg

    if model_name == "neural_network":
        constructor_cfg = dict(cfg)
        if "hidden_layer_sizes" in cfg and isinstance(cfg["hidden_layer_sizes"], list):
            constructor_cfg["hidden_layer_sizes"] = tuple(cfg["hidden_layer_sizes"])
        constructor_cfg.pop("epochs", None)
        constructor_cfg.setdefault("max_iter", cfg.get("epochs", cfg.get("max_iter", 300)))
        return MLPClassifier(**constructor_cfg), cfg

    valid_models = [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "neural_network",
    ]
    raise ValueError(f"Unsupported model_name '{model_name}'. Expected one of {valid_models}")


def evaluate_model(model, X_data, y_data):
    y_pred = model.predict(X_data)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_data)[:, 1]
    else:
        y_proba = y_pred

    return {
        "accuracy": float(accuracy_score(y_data, y_pred)),
        "precision": float(precision_score(y_data, y_pred, zero_division=0)),
        "recall": float(recall_score(y_data, y_pred, zero_division=0)),
        "f1": float(f1_score(y_data, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_data, y_proba)),
    }


def train_neural_network_with_progress(model, X_train, y_train, X_test, y_test, model_name, model_cfg):
    """Train MLPClassifier one epoch at a time so progress can be displayed."""
    epochs = int(model_cfg.get("epochs", model_cfg.get("max_iter", 300)))
    classes = np.unique(y_train)

    train_metrics = {}
    test_metrics = {}

    print(f"  Training {model_name} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            model.partial_fit(X_train, y_train, classes=classes)
        else:
            model.partial_fit(X_train, y_train)

        # Keep the loop informative without being too noisy.
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            current_loss = getattr(model, "loss_", None)
            print(f"    Epoch {epoch}/{epochs} - loss: {current_loss:.6f}" if current_loss is not None else f"    Epoch {epoch}/{epochs}")
            mlflow.log_metric("epoch_loss", float(current_loss) if current_loss is not None else 0.0, step=epoch)

    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)
    return train_metrics, test_metrics


def train_single_model(model_name=None, params_file="params.yaml"):
    params = load_params(params_file)

    if model_name is None:
        model_name = params["experiment"]["model_name"]

    features_path = Path(params["data"]["features_dir"]) / "selected"
    output_path = Path(params["data"]["models_dir"]) / "experiments"
    output_path.mkdir(parents=True, exist_ok=True)

    X_train = np.load(features_path / "X_train_selected.npy")
    y_train = pd.read_csv(features_path / "y_train.csv").iloc[:, 0]
    X_test = np.load(features_path / "X_test_selected.npy")
    y_test = pd.read_csv(features_path / "y_test.csv").iloc[:, 0]

    tracking_uri = params["experiment"].get("mlflow_tracking_uri", "file:./mlruns")
    experiment_name = params["experiment"].get("mlflow_experiment_name", "credit-risk-modeling")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    model, model_cfg = get_model(model_name, params)

    run_name = build_run_name(model_name, model_cfg)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_shape", str(X_train.shape))
        mlflow.log_param("test_shape", str(X_test.shape))

        for k, v in model_cfg.items():
            mlflow.log_param(f"{model_name}.{k}", sanitize_param_value(v))

        if model_name == "neural_network":
            train_metrics, test_metrics = train_neural_network_with_progress(
                model, X_train, y_train, X_test, y_test, model_name, model_cfg
            )
        else:
            print(f"  Training {model_name}...")
            model.fit(X_train, y_train)
            train_metrics = evaluate_model(model, X_train, y_train)
            test_metrics = evaluate_model(model, X_test, y_test)

        metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_file = output_path / "model.pkl"
        metrics_file = output_path / "metrics.json"
        run_info_file = output_path / "run_info.json"

        joblib.dump(model, model_file)

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        run_info = {
            "model_name": model_name,
            "mlflow_run_id": run.info.run_id,
            "mlflow_experiment": experiment_name,
            "tracking_uri": tracking_uri,
            "timestamp_utc": datetime.utcnow().isoformat(),
        }
        with open(run_info_file, "w") as f:
            json.dump(run_info, f, indent=2)

    print("   MLflow training complete")
    print(f"  Model: {model_name}")
    print(f"  Run name: {run_name}")
    print(f"  Run ID: {run.info.run_id}")
    print(f"  Train metrics: {train_metrics}")
    print(f"  Test metrics: {test_metrics}")
    print(f"  Saved model to: {model_file}")

    return model, metrics, run_info


def parse_args():
    parser = argparse.ArgumentParser(description="Train one model and log to MLflow")
    parser.add_argument("--model-name", type=str, default=None, help="Model name to train")
    parser.add_argument("--params-file", type=str, default="params.yaml", help="Path to params file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_single_model(model_name=args.model_name, params_file=args.params_file)
