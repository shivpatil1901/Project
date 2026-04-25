"""FastAPI inference service for credit default prediction."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import joblib
import time
from fastapi import FastAPI, HTTPException
from fastapi import Request, Response
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    customer: dict[str, Any] = Field(
        ...,
        description="Mapping of feature_name to input value for a single customer",
    )


class PredictResponse(BaseModel):
    default_probability: float
    predicted_class: int
    model_uri: str


class BatchPredictRequest(BaseModel):
    customers: list[dict[str, Any]] = Field(
        ...,
        description="List of customer feature mappings for batch inference",
    )


class BatchPredictItem(BaseModel):
    default_probability: float
    predicted_class: int


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPredictItem]
    model_uri: str


class ExplainRequest(BaseModel):
    customer: dict[str, Any] = Field(
        ...,
        description="Mapping of feature_name to input value for explainability",
    )
    top_k: int = Field(default=10, ge=1, le=100)


class ExplainFeatureItem(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    abs_shap_value: float


class ExplainResponse(BaseModel):
    model_uri: str
    default_probability: float
    base_value: float
    top_features: list[ExplainFeatureItem]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "selected" / "selected_features.csv"
DEFAULT_RUN_INFO_PATH = PROJECT_ROOT / "models" / "experiments" / "run_info.json"
DEFAULT_FEATURE_DICT_PATH = PROJECT_ROOT / "data" / "LCDataDictionary.xlsx"
DEFAULT_BACKGROUND_PATH = PROJECT_ROOT / "data" / "features" / "selected" / "X_train_selected.npy"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "data" / "features" / "selected" / "scaler.pkl"
DEFAULT_ENGINEERED_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "X_engineered.csv"


def normalize_feature_name(name: str) -> str:
    """Normalize feature names for robust matching across snake/camel case variants."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


class InferenceService:
    def __init__(self) -> None:
        self.feature_names = self._load_feature_names()
        self.feature_name_set = set(self.feature_names)
        self.model_uri = self._resolve_model_uri()
        self.model, self.sklearn_model = self._load_model(self.model_uri)
        self._align_feature_schema_with_model()
        self.expected_feature_count = self._infer_expected_feature_count()
        self.feature_descriptions = self._load_feature_descriptions()
        self.feature_defaults, self.binary_features = self._load_feature_defaults_and_binary()
        self.scaler_stats = self._load_scaler_stats()
        self.background_df = self._load_background_data()
        self._shap_explainer: shap.Explainer | None = None

    def _align_feature_schema_with_model(self) -> None:
        """Ensure inference feature ordering/count matches the loaded sklearn model."""
        model_feature_names = getattr(self.sklearn_model, "feature_names_in_", None)
        if model_feature_names is not None and len(model_feature_names) > 0:
            self.feature_names = [str(name) for name in model_feature_names]
            self.feature_name_set = set(self.feature_names)

    def _infer_expected_feature_count(self) -> int | None:
        """Infer the feature count required by the underlying model."""
        n_features = getattr(self.sklearn_model, "n_features_in_", None)
        if n_features is not None:
            return int(n_features)
        return None

    def _load_feature_names(self) -> list[str]:
        feature_list_path = Path(os.getenv("FEATURE_LIST_PATH", str(DEFAULT_FEATURES_PATH)))
        if not feature_list_path.exists():
            raise FileNotFoundError(f"Feature list not found: {feature_list_path}")

        feature_df = pd.read_csv(feature_list_path)
        if "selected_features" not in feature_df.columns:
            raise ValueError(
                "selected_features.csv must contain a 'selected_features' column"
            )
        return feature_df["selected_features"].tolist()

    def _resolve_model_uri(self) -> str:
        model_uri_override = os.getenv("MODEL_URI")
        if model_uri_override:
            return self._resolve_mlflow_identifier(model_uri_override)

        run_info_path = Path(os.getenv("RUN_INFO_PATH", str(DEFAULT_RUN_INFO_PATH)))
        if not run_info_path.exists():
            raise FileNotFoundError(
                f"run_info.json not found. Set MODEL_URI or RUN_INFO_PATH. Missing: {run_info_path}"
            )

        with open(run_info_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)

        run_id = run_info.get("mlflow_run_id")
        if not run_id:
            raise ValueError("run_info.json missing 'mlflow_run_id'")

        return self._resolve_mlflow_identifier(run_id)

    def _load_feature_descriptions(self) -> dict[str, str]:
        """Load optional feature descriptions from LCDataDictionary.xlsx."""
        dict_path = Path(os.getenv("FEATURE_DICT_PATH", str(DEFAULT_FEATURE_DICT_PATH)))
        if not dict_path.exists():
            return {}

        try:
            df = pd.read_excel(dict_path)
        except Exception:
            return {}

        if df.empty:
            return {}

        normalized_cols = {normalize_feature_name(c): c for c in df.columns}

        name_candidates = [
            "loanstatnew", "field", "feature", "name", "column", "variablename", "variable"
        ]
        desc_candidates = ["description", "desc", "explanation", "details", "definition"]

        name_col = next((normalized_cols[c] for c in name_candidates if c in normalized_cols), df.columns[0])
        desc_col = next((normalized_cols[c] for c in desc_candidates if c in normalized_cols), None)

        if desc_col is None:
            desc_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        raw_map: dict[str, str] = {}
        for _, row in df.iterrows():
            raw_name = str(row.get(name_col, "")).strip()
            if not raw_name or raw_name.lower() == "nan":
                continue
            raw_desc = str(row.get(desc_col, "")).strip()
            if raw_desc.lower() == "nan":
                raw_desc = ""
            raw_map[normalize_feature_name(raw_name)] = raw_desc

        descriptions: dict[str, str] = {}
        for feature in self.feature_names:
            descriptions[feature] = raw_map.get(normalize_feature_name(feature), "")

        return descriptions

    def _resolve_mlflow_identifier(self, value: str) -> str:
        """Resolve MODEL_URI override that may be a full URI, run ID, or experiment ID."""
        value = value.strip()
        if value.startswith(("runs:/", "models:/", "file:/", "s3:/", "http://", "https://")):
            return value

        # Direct local model path support (directory containing MLmodel).
        value_path = Path(value)
        if value_path.exists():
            return str(value_path)

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        # First try as run ID.
        try:
            client.get_run(value)
            return f"runs:/{value}/model"
        except MlflowException:
            pass

        # Then try as experiment ID and pick latest run.
        try:
            runs = client.search_runs(
                experiment_ids=[value],
                order_by=["attribute.start_time DESC"],
                max_results=50,
            )
        except MlflowException as exc:
            raise ValueError(
                f"Could not resolve MLflow identifier '{value}' as run_id or experiment_id"
            ) from exc

        if not runs:
            raise ValueError(
                f"No runs found for experiment '{value}'. Provide a valid run ID or full MODEL_URI."
            )

        # Prefer local model artifacts under mlruns/<exp_id>/models/*/artifacts when available.
        local_model_dir = self._latest_local_experiment_model_dir(value)
        if local_model_dir is not None:
            return str(local_model_dir)

        return f"runs:/{runs[0].info.run_id}/model"

    def _latest_local_experiment_model_dir(self, experiment_id: str) -> Path | None:
        """Find latest local model artifact directory for an MLflow experiment."""
        mlruns_root = PROJECT_ROOT / "mlruns" / experiment_id / "models"
        if not mlruns_root.exists():
            return None

        candidates = []
        for mlmodel_file in mlruns_root.glob("*/artifacts/MLmodel"):
            candidates.append(mlmodel_file)

        if not candidates:
            return None

        latest_mlmodel = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest_mlmodel.parent

    def _load_model(self, model_uri: str):
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        sklearn_model = None
        try:
            sklearn_model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            sklearn_model = None
        return pyfunc_model, sklearn_model

    def _load_feature_defaults_and_binary(self) -> tuple[dict[str, float], list[str]]:
        """Load realistic defaults (medians) and detect binary 0/1 flags from training features."""
        engineered_path = Path(
            os.getenv("ENGINEERED_FEATURES_PATH", str(DEFAULT_ENGINEERED_FEATURES_PATH))
        )
        defaults: dict[str, float] = {feature: 0.0 for feature in self.feature_names}
        binary_flags: list[str] = []

        if not engineered_path.exists():
            return defaults, binary_flags

        try:
            df = pd.read_csv(engineered_path)
            for feature in self.feature_names:
                if feature not in df.columns:
                    continue

                series = pd.to_numeric(df[feature], errors="coerce").dropna()
                if series.empty:
                    continue

                defaults[feature] = float(series.median())
                unique_vals = set(series.unique().tolist())
                if unique_vals.issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2:
                    binary_flags.append(feature)
        except Exception:
            return defaults, binary_flags

        return defaults, sorted(binary_flags)

    def _load_scaler_stats(self) -> dict[str, tuple[float, float]]:
        """Load per-feature mean/scale from training scaler for selected features."""
        scaler_path = Path(os.getenv("SCALER_PATH", str(DEFAULT_SCALER_PATH)))
        if not scaler_path.exists():
            return {}

        try:
            scaler = joblib.load(scaler_path)
            feature_names_in = getattr(scaler, "feature_names_in_", None)
            means = getattr(scaler, "mean_", None)
            scales = getattr(scaler, "scale_", None)
            if feature_names_in is None or means is None or scales is None:
                return {}

            stats_by_feature = {
                str(name): (float(mean), float(scale))
                for name, mean, scale in zip(feature_names_in, means, scales)
            }

            selected_stats: dict[str, tuple[float, float]] = {}
            for feature in self.feature_names:
                if feature in stats_by_feature:
                    selected_stats[feature] = stats_by_feature[feature]

            return selected_stats
        except Exception:
            return {}

    def _load_background_data(self) -> pd.DataFrame:
        background_path = Path(os.getenv("BACKGROUND_DATA_PATH", str(DEFAULT_BACKGROUND_PATH)))
        if background_path.exists():
            arr = np.load(background_path)
            if arr.ndim == 2 and arr.shape[1] == len(self.feature_names):
                sample_size = min(200, arr.shape[0])
                return pd.DataFrame(arr[:sample_size], columns=self.feature_names)

        # Fallback keeps API functional even if training artifacts are unavailable.
        return pd.DataFrame([np.zeros(len(self.feature_names))], columns=self.feature_names)

    def _predict_default_probability(self, X: pd.DataFrame) -> np.ndarray:
        if self.sklearn_model is not None and hasattr(self.sklearn_model, "predict_proba"):
            return np.asarray(self.sklearn_model.predict_proba(X)[:, 1], dtype=float)

        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)

        raw_preds = np.asarray(self.model.predict(X), dtype=float)
        return np.clip(raw_preds, 0.0, 1.0)

    def _prepare_model_input(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Match training-time standardization for selected features."""
        if not self.scaler_stats:
            X_scaled = X_raw.copy()
        else:
            X_scaled = X_raw.copy()
            for feature in self.feature_names:
                if feature not in self.scaler_stats:
                    continue
                mean, scale = self.scaler_stats[feature]
                safe_scale = scale if scale not in (0.0, -0.0) else 1.0
                X_scaled[feature] = (X_scaled[feature] - mean) / safe_scale

        # Defensive alignment: some legacy artifacts have selected_features.csv out-of-sync
        # with the trained model's expected dimensionality.
        if self.expected_feature_count is not None:
            current_count = X_scaled.shape[1]
            if current_count < self.expected_feature_count:
                for idx in range(self.expected_feature_count - current_count):
                    X_scaled[f"__auto_pad_{idx + 1}"] = 0.0
            elif current_count > self.expected_feature_count:
                X_scaled = X_scaled.iloc[:, : self.expected_feature_count]

        return X_scaled

    def _get_shap_explainer(self) -> shap.Explainer:
        if self._shap_explainer is None:
            def predict_fn(data: np.ndarray) -> np.ndarray:
                # SHAP already passes model-space inputs here; the background data
                # and the query points are both stored in the scaled feature space.
                X_model = pd.DataFrame(data, columns=self.feature_names)
                return self._predict_default_probability(X_model)

            self._shap_explainer = shap.Explainer(
                predict_fn,
                self.background_df,
                feature_names=self.feature_names,
                algorithm="permutation",
            )
        return self._shap_explainer

    def _coerce_numeric(self, value: Any) -> float:
        if isinstance(value, bool):
            return float(int(value))
        if value is None:
            return 0.0
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return 0.0
            return float(value)
        raise ValueError(f"Unsupported value type for numeric preprocessing: {type(value)}")

    def preprocess(self, payload: dict[str, Any]) -> pd.DataFrame:
        row: dict[str, float] = {}
        for feature in self.feature_names:
            row[feature] = self._coerce_numeric(payload.get(feature, 0.0))

        return pd.DataFrame([row], columns=self.feature_names)

    def compute_drift_score(self, payload: dict[str, Any]) -> float:
        """Compute a lightweight drift score against training medians."""
        X_raw = self.preprocess(payload)
        non_binary_features = [f for f in self.feature_names if f not in set(self.binary_features)]
        if not non_binary_features:
            return 0.0

        drifts: list[float] = []
        for feature in non_binary_features:
            observed = float(X_raw.iloc[0][feature])
            reference = float(self.feature_defaults.get(feature, 0.0))
            scale = abs(reference) + 1.0
            drifts.append(abs(observed - reference) / scale)

        if not drifts:
            return 0.0
        return float(np.mean(drifts))

    def predict(self, payload: dict[str, Any]) -> tuple[float, int]:
        X_raw = self.preprocess(payload)
        X = self._prepare_model_input(X_raw)
        prob_good = float(self._predict_default_probability(X)[0])
        # Flip to probability of default (model was trained with 1=good, 0=default)
        prob = 1.0 - prob_good
        pred_class = int(prob >= 0.5)
        return prob, pred_class

    def predict_batch(self, payloads: list[dict[str, Any]]) -> list[tuple[float, int]]:
        outputs: list[tuple[float, int]] = []
        for payload in payloads:
            outputs.append(self.predict(payload))
        return outputs

    def explain(self, payload: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
        X_raw = self.preprocess(payload)
        X = self._prepare_model_input(X_raw)
        prob_good = float(self._predict_default_probability(X)[0])
        # Flip to probability of default (model was trained with 1=good, 0=default)
        prob = 1.0 - prob_good

        explainer = self._get_shap_explainer()
        shap_result = explainer(X)

        shap_values = np.asarray(shap_result.values).reshape(-1)
        base_values = np.asarray(shap_result.base_values).reshape(-1)
        base_value_good = float(base_values[0]) if base_values.size > 0 else 0.0
        # Flip base value: when we flip prediction, base value also flips
        base_value = 1.0 - base_value_good

        details = []
        for idx, feature in enumerate(self.feature_names):
            feature_value = float(X_raw.iloc[0, idx])
            shap_value = float(shap_values[idx])
            # Negate SHAP values: when prediction flips, contributions flip sign
            flipped_shap_value = -shap_value
            details.append(
                {
                    "feature": feature,
                    "feature_value": feature_value,
                    "shap_value": flipped_shap_value,
                    "abs_shap_value": abs(flipped_shap_value),
                }
            )

        details.sort(key=lambda item: item["abs_shap_value"], reverse=True)
        return {
            "model_uri": self.model_uri,
            "default_probability": prob,
            "base_value": base_value,
            "top_features": details[:top_k],
        }


app = FastAPI(title="Credit Risk Inference API", version="1.0.0")
service = InferenceService()


# Prometheus monitoring metrics
REQUEST_COUNT = Counter(
    "credit_risk_api_requests_total",
    "Total count of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "credit_risk_api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

BUSINESS_LATENCY = Histogram(
    "credit_risk_business_latency_seconds",
    "Business logic latency for prediction/explainability endpoints",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

DRIFT_SCORE = Gauge(
    "credit_risk_feature_drift_score",
    "Mean normalized absolute deviation from training medians for latest scored request",
)

DRIFT_SCORE_HIST = Histogram(
    "credit_risk_feature_drift_score_distribution",
    "Distribution of drift scores over scored requests",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    # Skip /metrics self-observation to avoid noisy metric recursion.
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start_time

    endpoint = request.url.path
    method = request.method
    status_code = str(response.status_code)

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)

    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_uri": service.model_uri,
        "required_features": service.feature_names,
        "expected_feature_count": service.expected_feature_count,
        "feature_descriptions": service.feature_descriptions,
        "feature_defaults": service.feature_defaults,
        "binary_features": service.binary_features,
        "required_feature_count": len(service.feature_names),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        op_start = time.perf_counter()
        drift = service.compute_drift_score(request.customer)
        prob, pred_class = service.predict(request.customer)
        BUSINESS_LATENCY.labels(operation="predict").observe(time.perf_counter() - op_start)
        DRIFT_SCORE.set(drift)
        DRIFT_SCORE_HIST.observe(drift)
        return PredictResponse(
            default_probability=prob,
            predicted_class=pred_class,
            model_uri=service.model_uri,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    try:
        op_start = time.perf_counter()
        predictions = service.predict_batch(request.customers)
        BUSINESS_LATENCY.labels(operation="predict_batch").observe(time.perf_counter() - op_start)

        if request.customers:
            drifts = [service.compute_drift_score(customer) for customer in request.customers]
            avg_drift = float(np.mean(drifts)) if drifts else 0.0
            DRIFT_SCORE.set(avg_drift)
            for drift in drifts:
                DRIFT_SCORE_HIST.observe(drift)

        return BatchPredictResponse(
            predictions=[
                BatchPredictItem(default_probability=prob, predicted_class=pred_class)
                for prob, pred_class in predictions
            ],
            model_uri=service.model_uri,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    try:
        op_start = time.perf_counter()
        drift = service.compute_drift_score(request.customer)
        result = service.explain(request.customer, top_k=request.top_k)
        BUSINESS_LATENCY.labels(operation="explain").observe(time.perf_counter() - op_start)
        DRIFT_SCORE.set(drift)
        DRIFT_SCORE_HIST.observe(drift)
        return ExplainResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explainability failed: {exc}") from exc
