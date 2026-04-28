"""Streamlit frontend for credit default probability prediction."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9090")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://127.0.0.1:3000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://127.0.0.1:5000")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://127.0.0.1:8080")
AIRFLOW_PUBLIC_URL = os.getenv("AIRFLOW_PUBLIC_URL", "http://localhost:8081")
AIRFLOW_DAG_ID = os.getenv("AIRFLOW_DAG_ID", "credit_risk_data_ingestion")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def fetch_health() -> dict[str, Any]:
    response = requests.get(f"{BACKEND_URL}/health", timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_feature_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def remap_columns_to_required(df: pd.DataFrame, required_features: list[str]) -> pd.DataFrame:
    required_by_norm = {normalize_feature_name(f): f for f in required_features}
    mapped_columns: dict[str, str] = {}
    extra_columns: list[str] = []

    for column in df.columns:
        norm_col = normalize_feature_name(column)
        if norm_col in required_by_norm:
            mapped_columns[column] = required_by_norm[norm_col]
        else:
            extra_columns.append(column)

    if extra_columns:
        st.warning(
            "Ignoring non-training columns in uploaded CSV: " + ", ".join(extra_columns)
        )

    remapped = df.rename(columns=mapped_columns)
    missing = [f for f in required_features if f not in remapped.columns]
    if missing:
        raise ValueError("CSV is missing required training features: " + ", ".join(missing))

    return remapped[required_features].copy()


def _service_health(url: str, endpoint: str = "/") -> tuple[bool, str]:
    # Airflow 3 moved health to /api/v2/monitor/health; keep backward fallback.
    if endpoint == "/health":
        try:
            response = requests.get(f"{url.rstrip('/')}/api/v2/monitor/health", timeout=4)
            response.raise_for_status()
            return True, "reachable"
        except Exception:
            pass

    try:
        response = requests.get(f"{url.rstrip('/')}{endpoint}", timeout=4)
        # Some services (for example MLflow behind stricter defaults) may return
        # 401/403 for unauthenticated root requests while still being healthy.
        if response.status_code in (401, 403):
            return True, f"reachable ({response.status_code})"
        response.raise_for_status()
        return True, "reachable"
    except Exception as exc:
        return False, str(exc)


def _prom_query(query: str) -> float | None:
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", {}).get("result", [])
        if not data:
            return None
        value = data[0].get("value", [None, None])[1]
        return float(value) if value is not None else None
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_airflow_dag_runs(limit: int = 20) -> pd.DataFrame:
    endpoints = [
        f"{AIRFLOW_URL}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns",
        f"{AIRFLOW_URL}/api/v2/dags/{AIRFLOW_DAG_ID}/dagRuns",
    ]

    for endpoint in endpoints:
        for auth in [(AIRFLOW_USER, AIRFLOW_PASSWORD), None]:
            try:
                response = requests.get(
                    endpoint,
                    params={"limit": limit, "order_by": "-start_date"},
                    auth=auth,
                    timeout=4,
                )
                response.raise_for_status()
                payload = response.json()
                runs = payload.get("dag_runs", []) if isinstance(payload, dict) else []
                if not runs:
                    continue
                df = pd.DataFrame(runs)
                keep_cols = [
                    "dag_run_id",
                    "state",
                    "logical_date",
                    "start_date",
                    "end_date",
                    "run_type",
                ]
                return df[[c for c in keep_cols if c in df.columns]].copy()
            except Exception:
                continue

    return pd.DataFrame()


def _read_expected_mlflow_experiment_name() -> str | None:
    run_info_path = PROJECT_ROOT / "models" / "experiments" / "run_info.json"
    if not run_info_path.exists():
        return None

    try:
        with open(run_info_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        name = payload.get("mlflow_experiment")
        return str(name).strip() if name else None
    except Exception:
        return None


def _discover_mlflow_experiment_ids() -> list[str]:
    ids: list[str] = []
    exp_name = _read_expected_mlflow_experiment_name()

    if exp_name:
        try:
            response = requests.get(
                f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": exp_name},
                timeout=12,
            )
            response.raise_for_status()
            exp = response.json().get("experiment", {})
            exp_id = exp.get("experiment_id")
            if exp_id is not None:
                ids.append(str(exp_id))
        except Exception:
            pass

    try:
        response = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            json={"max_results": 200},
            timeout=12,
        )
        response.raise_for_status()
        experiments = response.json().get("experiments", [])
        for exp in experiments:
            exp_id = exp.get("experiment_id")
            if exp_id is not None:
                ids.append(str(exp_id))
    except Exception:
        pass

    if not ids:
        ids = ["0"]

    # Keep insertion order while de-duplicating.
    return list(dict.fromkeys(ids))


def _fetch_mlflow_runs(max_results: int = 20) -> pd.DataFrame:
    payload = {
        "max_results": max_results,
        "order_by": ["attribute.start_time DESC"],
    }

    experiment_ids = _discover_mlflow_experiment_ids()
    rows: list[dict[str, Any]] = []

    for exp_id in experiment_ids:
        try:
            exp_payload = dict(payload)
            exp_payload["experiment_ids"] = [exp_id]
            response = requests.post(
                f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
                json=exp_payload,
                timeout=12,
            )
            response.raise_for_status()
            runs = response.json().get("runs", [])
            for run in runs:
                info = run.get("info", {})
                metrics = run.get("data", {}).get("metrics", [])
                metrics_map = {m.get("key"): m.get("value") for m in metrics}
                rows.append(
                    {
                        "experiment_id": exp_id,
                        "run_id": info.get("run_id", ""),
                        "status": info.get("status", ""),
                        "start_time": info.get("start_time", ""),
                        "end_time": info.get("end_time", ""),
                        "roc_auc": metrics_map.get("roc_auc"),
                        "f1_score": metrics_map.get("f1_score"),
                        "accuracy": metrics_map.get("accuracy"),
                    }
                )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "start_time" in df.columns:
        df = df.sort_values(by="start_time", ascending=False, kind="stable")
    return df.head(max_results).reset_index(drop=True)


def _run_dvc_command(command: str) -> tuple[int, str]:
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        shell=True,
        capture_output=True,
        text=True,
    )
    output = (process.stdout or "") + ("\n" + process.stderr if process.stderr else "")
    return process.returncode, output.strip()


def render_prediction_studio() -> None:
    st.title("Credit Default Probability Estimator")
    st.caption("Enter customer feature values and predict probability of default.")

    try:
        health = fetch_health()
    except Exception as exc:
        st.error(f"Could not connect to backend at {BACKEND_URL}: {exc}")
        st.stop()

    feature_names = health.get("required_features", [])
    feature_descriptions = health.get("feature_descriptions", {})
    feature_defaults = health.get("feature_defaults", {})
    binary_features = set(health.get("binary_features", []))
    st.success(
        f"Backend is healthy. Loaded model from: {health.get('model_uri', 'unknown')}"
    )

    tab_single, tab_csv = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

    with tab_single:
        st.subheader("Customer Feature Inputs")
        with st.form("predict_form"):
            inputs: dict[str, float] = {}

            cols = st.columns(3)
            for idx, feature in enumerate(feature_names):
                with cols[idx % 3]:
                    desc = feature_descriptions.get(feature, "")
                    default_value = float(feature_defaults.get(feature, 0.0))
                    if feature in binary_features:
                        inputs[feature] = float(
                            st.selectbox(
                                label=feature,
                                options=[0, 1],
                                index=1 if default_value >= 0.5 else 0,
                                help=desc if desc else None,
                            )
                        )
                    else:
                        inputs[feature] = st.number_input(
                            label=feature,
                            value=default_value,
                            format="%.6f",
                            help=desc if desc else None,
                        )

            submitted = st.form_submit_button("Predict Default Probability")

        if submitted:
            non_binary_features = [f for f in feature_names if f not in binary_features]
            zero_count = sum(1 for f in non_binary_features if abs(float(inputs.get(f, 0.0))) < 1e-12)
            if non_binary_features and zero_count >= max(5, int(0.5 * len(non_binary_features))):
                st.warning(
                    f"{zero_count} non-binary fields are set to 0. This can represent an unrealistic profile and may lead to unstable predictions. "
                    "Consider using training-median-like values for missing fields."
                )

            payload = {"customer": inputs}
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json=payload,
                    timeout=60,
                )
                if response.status_code != 200:
                    st.error(f"Prediction failed: {response.text}")
                    st.stop()

                result = response.json()
                prob = float(result["default_probability"])
                pred_class = int(result["predicted_class"])

                st.subheader("Prediction Result")
                st.metric("Probability of Default", f"{prob:.4f}")
                st.metric("Predicted Class (>=0.5 is default)", pred_class)

                result_df = pd.DataFrame(
                    [{
                        "default_probability": prob,
                        "predicted_class": pred_class,
                        "model_uri": result.get("model_uri", ""),
                    }]
                )
                st.dataframe(result_df, use_container_width=True)

                st.session_state["last_inputs"] = inputs

            except Exception as exc:
                st.error(f"Request error: {exc}")

        st.subheader("SHAP Explainability")

        with st.expander("📚 How to Interpret SHAP Values", expanded=False):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** explains why your model made a specific prediction.

            **Key Concepts:**
            - **Base Value**: The average default probability across all training customers (~13% for this model). This is your starting point.
            - **SHAP Value per Feature**: How much that feature's actual value increases (+) or decreases (−) the final prediction.

            **The Formula:**
            ```
            Final Prediction = Base Value + Sum of (SHAP Values for all features)
            ```

            **Example:**
            - Base Value: 0.13 (13% average default probability)
            - annual_inc = $80k -> SHAP: -0.02 (high income reduces default risk by 2%)
            - dti = 0.45 -> SHAP: +0.03 (high debt ratio increases default risk by 3%)
            - int_rate = 12% -> SHAP: +0.02 (elevated rate increases default risk by 2%)
            - **Final: 0.13 - 0.02 + 0.03 + 0.02 = 0.16** (16% predicted default probability)

            **Interpreting the Bar Chart:**
            - **Longer red bars** = Features increasing default risk
            - **Longer green bars** = Features decreasing default risk
            - **Bar length** = Magnitude of impact on this specific prediction
            """)

        top_k = st.slider("Top features to explain", min_value=5, max_value=25, value=10, step=1)
        explain_button = st.button("Explain Last Prediction")

        if explain_button:
            if "last_inputs" not in st.session_state:
                st.warning("Run a prediction first to generate SHAP explanations.")
            else:
                explain_payload = {
                    "customer": st.session_state["last_inputs"],
                    "top_k": top_k,
                }
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/explain",
                        json=explain_payload,
                        timeout=180,
                    )
                    if response.status_code != 200:
                        st.error(f"Explainability failed: {response.text}")
                        st.stop()

                    explain_result = response.json()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("SHAP Base Value", f"{float(explain_result['base_value']):.4f}")
                        st.caption("Average default probability across training data (~13%)")
                    with col2:
                        st.metric("Explained Default Probability", f"{float(explain_result['default_probability']):.4f}")
                        st.caption("Final model prediction = base value + sum of SHAP values")

                    st.subheader("Top Contributing Features")
                    st.info("""
                    **How to read this table:**
                    - **feature**: The customer feature (e.g., annual_inc, dti)
                    - **feature_value**: The actual value of this feature for the customer
                    - **shap_value**: How much this feature changed the prediction (+/- direction)
                    - **abs_shap_value**: Magnitude of impact (for sorting by importance)

                    Positive SHAP values increase default risk; negative SHAP values decrease default risk.
                    """)

                    shap_df = pd.DataFrame(explain_result["top_features"])
                    st.dataframe(shap_df, use_container_width=True)

                    st.subheader("Feature Impact Visualization")

                    colors = ["#d62728" if x > 0 else "#2ca02c" for x in shap_df["shap_value"]]
                    fig = go.Figure(data=[
                        go.Bar(
                            x=shap_df["shap_value"],
                            y=shap_df["feature"],
                            orientation="h",
                            marker=dict(color=colors),
                            text=[f"{v:.4f}" for v in shap_df["shap_value"]],
                            textposition="outside",
                        )
                    ])
                    fig.update_layout(
                        title="SHAP Feature Impact (Contribution to Default Probability)",
                        xaxis_title="SHAP Value",
                        yaxis_title="Feature",
                        height=400,
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Red = Increases default risk | Green = Decreases default risk | Bar length = Magnitude")

                except Exception as exc:
                    st.error(f"SHAP request error: {exc}")

    with tab_csv:
        st.subheader("Upload CSV for Batch Scoring")
        st.caption(
            "Column names are mapped case-insensitively across styles like annualInc -> annual_inc. "
            "Only train-time features are allowed."
        )

        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
                mapped_df = remap_columns_to_required(input_df, feature_names)
                st.success(f"CSV validated. Rows: {len(mapped_df)}")
                st.dataframe(mapped_df.head(20), use_container_width=True)

                if st.button("Predict Batch Default Probabilities"):
                    payload = {"customers": mapped_df.to_dict(orient="records")}
                    response = requests.post(
                        f"{BACKEND_URL}/predict-batch",
                        json=payload,
                        timeout=120,
                    )
                    if response.status_code != 200:
                        st.error(f"Batch prediction failed: {response.text}")
                        st.stop()

                    result = response.json()
                    preds = result.get("predictions", [])
                    pred_df = pd.DataFrame(preds)
                    output_df = pd.concat([mapped_df.reset_index(drop=True), pred_df], axis=1)
                    st.subheader("Batch Prediction Results")
                    st.dataframe(output_df, use_container_width=True)
                    st.download_button(
                        "Download Predictions CSV",
                        output_df.to_csv(index=False).encode("utf-8"),
                        file_name="credit_default_predictions.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"CSV validation/prediction error: {exc}")


def render_pipeline_ops_console() -> None:
    st.title("Pipeline Ops Console")
    st.caption(
        "Unified observability and control panel for ingestion, training, serving, and monitoring across Airflow, DVC, MLflow, Prometheus, and Grafana."
    )

    refresh_seconds = st.sidebar.slider("Auto-refresh interval (seconds)", min_value=0, max_value=120, value=0, step=5)
    if refresh_seconds > 0:
        st.caption(f"Auto-refresh enabled every {refresh_seconds} seconds. Use browser refresh if you need immediate update.")

    services = [
        ("API", BACKEND_URL, "/health"),
        ("Prometheus", PROMETHEUS_URL, "/-/healthy"),
        ("Grafana", GRAFANA_URL, "/api/health"),
        ("MLflow", MLFLOW_URL, "/"),
        ("Airflow", AIRFLOW_URL, "/health"),
    ]

    st.subheader("Platform Health")
    cols = st.columns(len(services))
    for col, (name, url, endpoint) in zip(cols, services):
        with col:
            ok, detail = _service_health(url, endpoint=endpoint)
            st.metric(name, "UP" if ok else "DOWN")
            if ok:
                st.caption(url)
            else:
                st.caption(detail[:90])

    st.markdown("---")
    st.subheader("Pipeline Management")
    m1, m2, m3 = st.columns(3)
    with m1:
        if st.button("Run DVC Pipeline (dvc repro)", use_container_width=True):
            with st.spinner("Running dvc repro..."):
                code, output = _run_dvc_command("dvc repro")
            st.code(output[-8000:] if output else "No output")
            if code == 0:
                st.success("Pipeline run completed successfully.")
            else:
                st.error(f"Pipeline run failed with exit code {code}.")

    with m2:
        if st.button("Run MLflow Experiment (DVC)", use_container_width=True):
            with st.spinner("Running dvc exp run train_model_mlflow..."):
                code, output = _run_dvc_command("dvc exp run train_model_mlflow")
            st.code(output[-8000:] if output else "No output")
            if code == 0:
                st.success("Experiment run completed successfully.")
            else:
                st.error(f"Experiment failed with exit code {code}.")

    with m3:
        if st.button("Check DVC Status", use_container_width=True):
            with st.spinner("Checking dvc status..."):
                code, output = _run_dvc_command("dvc status")
            st.code(output[-5000:] if output else "No output")
            if code == 0:
                st.success("DVC status check finished.")
            else:
                st.warning("DVC status check returned warnings/errors.")

    st.markdown("---")
    st.subheader("Error, Failure, and Success Tracking")

    airflow_ok, _ = _service_health(AIRFLOW_URL, endpoint="/health")
    airflow_runs = _fetch_airflow_dag_runs(limit=25) if airflow_ok else pd.DataFrame()
    mlflow_runs = _fetch_mlflow_runs(max_results=25)

    a_success = int((airflow_runs["state"] == "success").sum()) if not airflow_runs.empty and "state" in airflow_runs.columns else 0
    a_failed = int((airflow_runs["state"] == "failed").sum()) if not airflow_runs.empty and "state" in airflow_runs.columns else 0
    a_running = int((airflow_runs["state"].isin(["queued", "running"]).sum())) if not airflow_runs.empty and "state" in airflow_runs.columns else 0

    m_success = int((mlflow_runs["status"] == "FINISHED").sum()) if not mlflow_runs.empty and "status" in mlflow_runs.columns else 0
    m_failed = int((mlflow_runs["status"] == "FAILED").sum()) if not mlflow_runs.empty and "status" in mlflow_runs.columns else 0
    m_running = int((mlflow_runs["status"] == "RUNNING").sum()) if not mlflow_runs.empty and "status" in mlflow_runs.columns else 0

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Airflow Success", a_success)
    s2.metric("Airflow Failed", a_failed)
    s3.metric("Airflow Running", a_running)
    s4.metric("MLflow Success", m_success)
    s5.metric("MLflow Failed", m_failed)
    s6.metric("MLflow Running", m_running)

    detail_tab1, detail_tab2 = st.tabs(["Airflow DAG Runs", "MLflow Runs"])
    with detail_tab1:
        if airflow_runs.empty:
            st.info("No Airflow run data found. Ensure Airflow API is reachable and credentials are valid.")
        else:
            st.dataframe(airflow_runs, use_container_width=True)

    with detail_tab2:
        if mlflow_runs.empty:
            st.info("No MLflow run data found. Ensure MLflow tracking server is reachable.")
        else:
            st.dataframe(mlflow_runs, use_container_width=True)

    st.markdown("---")
    st.subheader("Speed and Throughput")
    throughput = _prom_query("sum(rate(credit_risk_api_requests_total[5m]))")
    p95_latency = _prom_query(
        "histogram_quantile(0.95, sum(rate(credit_risk_api_request_latency_seconds_bucket[5m])) by (le))"
    )
    error_rate = _prom_query('sum(rate(credit_risk_api_requests_total{status_code=~"5.."}[5m]))')
    avg_drift = _prom_query("avg(credit_risk_feature_drift_score)")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Req/sec (5m)", f"{throughput:.3f}" if throughput is not None else "N/A")
    t2.metric("P95 Latency (s)", f"{p95_latency:.4f}" if p95_latency is not None else "N/A")
    t3.metric("5xx Error Rate", f"{error_rate:.4f}" if error_rate is not None else "N/A")
    t4.metric("Avg Drift Score", f"{avg_drift:.4f}" if avg_drift is not None else "N/A")

    if throughput is None and p95_latency is None:
        st.warning("Prometheus metrics unavailable. Verify Prometheus is scraping your API /metrics endpoint.")

    st.markdown("---")
    st.subheader("Tool Navigation")
    st.markdown(
        f"""
        - Airflow UI: {AIRFLOW_PUBLIC_URL}
        - MLflow UI: {MLFLOW_URL}
        - Prometheus UI: {PROMETHEUS_URL}
        - Grafana UI: {GRAFANA_URL}
        """
    )

    if st.button("Inspect Last MLflow Run Metadata", use_container_width=True):
        run_info_path = PROJECT_ROOT / "models" / "experiments" / "run_info.json"
        if run_info_path.exists():
            try:
                with open(run_info_path, "r", encoding="utf-8") as f:
                    run_info = json.load(f)
                st.json(run_info)
            except Exception as exc:
                st.error(f"Failed to read run_info.json: {exc}")
        else:
            st.warning("run_info.json not found. Run train_model_mlflow stage first.")


def main() -> None:
    st.set_page_config(page_title="Credit Risk Platform Console", page_icon="📊", layout="wide")
    screen = st.sidebar.radio("Choose Screen", ["Prediction Studio", "Pipeline Ops Console"])
    if screen == "Prediction Studio":
        render_prediction_studio()
    else:
        render_pipeline_ops_console()


if __name__ == "__main__":
    main()
