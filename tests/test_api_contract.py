from __future__ import annotations

import pytest
import pandas as pd

from api import main as api_main


def _make_service() -> api_main.InferenceService:
    service = api_main.InferenceService.__new__(api_main.InferenceService)
    service.feature_names = ["emp_length", "annual_inc", "debt_settlement_flag"]
    service.feature_name_set = set(service.feature_names)
    service.binary_features = ["debt_settlement_flag"]
    service.feature_defaults = {
        "emp_length": 2.0,
        "annual_inc": 50000.0,
        "debt_settlement_flag": 0.0,
    }
    service.scaler_stats = {}
    service.model_uri = "models:/dummy"
    service.feature_descriptions = {}
    service.background_df = pd.DataFrame([[0.0, 0.0, 0.0]], columns=service.feature_names)
    service._shap_explainer = None

    def _predict_default_probability(X):
        return __import__("numpy").array([0.2])

    service._predict_default_probability = _predict_default_probability  # type: ignore[attr-defined]
    return service


def test_preprocess_fills_missing_features_and_rejects_unknowns():
    service = _make_service()

    frame = service.preprocess({"emp_length": 5, "debt_settlement_flag": 1})

    assert list(frame.columns) == service.feature_names
    assert frame.iloc[0]["annual_inc"] == 0.0
    assert frame.iloc[0]["debt_settlement_flag"] == 1.0

    with pytest.raises(ValueError):
        service.preprocess({"unknown_feature": 1})


def test_compute_drift_score_ignores_binary_flags():
    service = _make_service()

    drift = service.compute_drift_score(
        {"emp_length": 5, "annual_inc": 50000.0, "debt_settlement_flag": 1}
    )

    assert drift == pytest.approx(0.5)


def test_normalize_feature_name_collapses_case_and_symbols():
    assert api_main.normalize_feature_name("Annual Inc %") == "annualinc"
