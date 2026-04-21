from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.process import (
    create_target_variable,
    load_data,
    prepare_numeric_features,
    process_categorical_and_dates,
    remove_missing_columns,
)
from features.selection import (
    remove_leaky_features,
    scale_features,
    select_features_with_rf,
    train_test_split_data,
)
from model.evaluate import evaluate_model as evaluate_single_model
from model.train_mlflow import build_run_name, get_model, sanitize_param_value


def test_create_target_variable_maps_labels_and_drops_unknowns():
    frame = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Unknown Status"],
            "feature_a": [1, 2, 3],
        }
    )

    result = create_target_variable(frame)

    assert list(result["loan_status_binary"]) == [1, 0]
    assert "loan_status" not in result.columns
    assert len(result) == 2


def test_load_data_reads_input_csv(tmp_path):
    raw_path = tmp_path / "loan_data.csv"

    raw_frame = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "value": [1, 2, 3],
        }
    )
    raw_frame.to_csv(raw_path, index=False)

    loaded = pd.DataFrame(load_data(raw_path))
    assert loaded.equals(raw_frame)


def test_remove_missing_columns_drops_columns_above_threshold():
    frame = pd.DataFrame(
        {
            "keep": [1, 2, 3, 4],
            "drop_me": [np.nan, np.nan, np.nan, 1],
            "also_keep": [10, 11, 12, 13],
        }
    )

    result = remove_missing_columns(frame, threshold=50)

    assert "drop_me" not in result.columns
    assert "keep" in result.columns
    assert "also_keep" in result.columns


def test_process_categorical_and_dates_creates_expected_flags_and_numeric_columns():
    frame = pd.DataFrame(
        {
            "issue_d": ["Jan-2020", "Feb-2020"],
            "earliest_cr_line": ["Jan-2010", "Jan-2012"],
            "last_pymnt_d": ["Jan-2021", None],
            "last_credit_pull_d": ["Feb-2021", None],
            "int_rate": ["10.5%", "12.0%"],
            "revol_util": ["25%", "30%"],
            "debt_settlement_flag": ["Y", "N"],
            "term": ["36 months", "60 months"],
            "emp_length": ["10+ years", "< 1 year"],
            "hardship_flag": ["Y", "N"],
            "loan_status_binary": [1, 0],
        }
    )

    result = process_categorical_and_dates(frame)

    assert "term" not in result.columns
    assert "term_36_months" in result.columns
    assert "int_rate%" in result.columns
    assert "revol_util%" in result.columns
    assert result["debt_settlement_flag"].tolist() == [1, 0]
    assert result["term_36_months"].tolist() == [1, 0]


def test_prepare_numeric_features_imputes_and_returns_target_series():
    frame = pd.DataFrame(
        {
            "loan_status_binary": [1, 0],
            "emp_length": [10, 2],
            "annual_inc": [100000.0, np.nan],
            "dti": [0.1, 0.2],
            "sub_grade": ["A1", "B2"],
            "application_type": ["Individual", "Joint App"],
            "initial_list_status": ["w", "f"],
            "addr_state": ["CA", "NY"],
            "purpose": ["debt_consolidation", "credit_card"],
            "home_ownership": ["MORTGAGE", "RENT"],
        }
    )

    X, y = prepare_numeric_features(frame)

    assert list(y) == [1, 0]
    assert not X.isna().any().any()
    assert any(col.startswith("home_ownership_") for col in X.columns)


def test_remove_leaky_features_and_split_scale_and_rf_selection():
    rng = np.random.default_rng(7)
    y = pd.Series([0, 1] * 40)
    signal = y.to_numpy().astype(float) * 5 + rng.normal(0, 0.05, size=len(y))
    noise = rng.normal(0, 1, size=len(y))
    leaky = np.arange(len(y))
    frame = pd.DataFrame(
        {
            "signal": signal,
            "noise": noise,
            "total_rec_prncp": leaky,
        }
    )

    cleaned, _ = remove_leaky_features(frame, ["total_rec_prncp"])
    assert "total_rec_prncp" not in cleaned.columns

    X_train, X_test, y_train, y_test = train_test_split_data(
        cleaned, y, test_size=0.25, random_state=42
    )
    assert len(X_train) + len(X_test) == len(frame)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape[1] == 2
    assert scaler is not None

    X_train_selected, X_test_selected, selected_features, rf_model, selector = select_features_with_rf(
        X_train_scaled,
        X_test_scaled,
        y_train,
        X_train.columns,
        n_estimators=50,
        threshold="mean",
        random_state=42,
    )

    assert X_train_selected.shape[0] == X_train.shape[0]
    assert X_test_selected.shape[0] == X_test.shape[0]
    assert "signal" in list(selected_features)
    assert rf_model is not None
    assert selector is not None


def test_sanitize_param_value_and_build_run_name_are_stable():
    assert sanitize_param_value({"a": 1}) == json.dumps({"a": 1})
    assert build_run_name("rf", {"b": 2, "a": 1}) == "train_rf_a=1_b=2"


def test_get_model_handles_logistic_regression_and_neural_network():
    params = {
        "models": {
            "logistic_regression": {"random_state": 99},
            "neural_network": {
                "hidden_layer_sizes": [32, 16],
                "epochs": 5,
                "random_state": 7,
                "solver": "adam",
            },
        }
    }

    lr_model, lr_cfg = get_model("logistic_regression", params)
    nn_model, nn_cfg = get_model("neural_network", params)

    assert lr_model.max_iter == 1000
    assert lr_cfg["random_state"] == 99
    assert tuple(nn_model.hidden_layer_sizes) == (32, 16)
    assert nn_model.max_iter == 5
    assert nn_cfg["epochs"] == 5


class _DummyClassifier:
    def predict(self, X):
        return np.array([0, 1, 1, 0])

    def predict_proba(self, X):
        return np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.85, 0.15],
            ]
        )


def test_evaluate_model_computes_expected_metrics():
    X = np.zeros((4, 2))
    y = pd.Series([0, 1, 1, 0])

    metrics, class_report = evaluate_single_model(_DummyClassifier(), X, y, "Dummy")

    assert metrics["Accuracy"] == pytest.approx(1.0)
    assert metrics["Precision"] == pytest.approx(1.0)
    assert metrics["Recall"] == pytest.approx(1.0)
    assert metrics["TN"] == 2
    assert metrics["TP"] == 2
    assert class_report["accuracy"] == pytest.approx(1.0)
