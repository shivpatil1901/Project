"""Streamlit frontend for credit default probability prediction."""
from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


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
        raise ValueError(
            "CSV contains unknown/non-training features: " + ", ".join(extra_columns)
        )

    remapped = df.rename(columns=mapped_columns)
    missing = [f for f in required_features if f not in remapped.columns]
    if missing:
        raise ValueError("CSV is missing required training features: " + ", ".join(missing))

    return remapped[required_features].copy()


def main() -> None:
    st.set_page_config(page_title="Credit Risk Predictor", page_icon="📉", layout="wide")
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
            - annual_inc = $80k → SHAP: −0.02 (high income reduces default risk by 2%)
            - dti = 0.45 → SHAP: +0.03 (high debt ratio increases default risk by 3%)
            - int_rate = 12% → SHAP: +0.02 (elevated rate increases default risk by 2%)
            - **Final: 0.13 − 0.02 + 0.03 + 0.02 = 0.16** (16% predicted default probability)
            
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
                    - **shap_value**: How much this feature changed the prediction (± direction)
                    - **abs_shap_value**: Magnitude of impact (for sorting by importance)
                    
                    Positive SHAP values ➜ Increase default risk | Negative SHAP values ➜ Decrease default risk
                    """)
                    
                    shap_df = pd.DataFrame(explain_result["top_features"])
                    st.dataframe(shap_df, use_container_width=True)

                    st.subheader("Feature Impact Visualization")
                    
                    # Create colored bar chart: red for positive (risk increase), green for negative (risk decrease)
                    colors = ["#d62728" if x > 0 else "#2ca02c" for x in shap_df["shap_value"]]
                    fig = go.Figure(data=[
                        go.Bar(
                            x=shap_df["shap_value"],
                            y=shap_df["feature"],
                            orientation='h',
                            marker=dict(color=colors),
                            text=[f"{v:.4f}" for v in shap_df["shap_value"]],
                            textposition='outside',
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
                    st.caption("🔴 Red = Increases default risk | 🟢 Green = Decreases default risk | Bar length = Magnitude of impact")

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


if __name__ == "__main__":
    main()
