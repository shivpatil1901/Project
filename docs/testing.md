# Test Plan, Test Cases, and Report

## 1. Test Plan

### Objective

Validate the pipeline utilities, model evaluation helpers, MLflow model factory, and API contract behavior using unit tests.

### Scope

- Data processing helper functions
- Feature selection helpers
- MLflow model factory helpers
- Evaluation metric computation
- API preprocessing and drift computation logic

### Acceptance Criteria

- All unit tests pass.
- Core preprocessing functions preserve expected label mapping.
- Feature selection helpers remove leakage-prone columns correctly.
- API contract validation rejects unknown features and handles missing fields safely.
- Drift computation excludes binary flags.

## 2. Enlisted Test Cases

| ID | Test Case | Expected Result |
|---|---|---|
| TC-01 | Target mapping in `create_target_variable` | Good loans map to `1`, default-like labels to `0`, unknown labels dropped |
| TC-02 | Missing-column removal | Columns above missing threshold are removed |
| TC-03 | Date/categorical processing | Flags and engineered date columns are created correctly |
| TC-04 | Numeric prep and imputation | Output has no missing values and target series is preserved |
| TC-05 | Leakage removal and split/scaling | Leaky columns removed; split and scaling succeed |
| TC-06 | RF feature selection | Informative feature is selected |
| TC-07 | MLflow helper functions | Run-name ordering and model factory behavior are correct |
| TC-08 | Model evaluation metrics | Metrics are computed correctly for a dummy perfect classifier |
| TC-09 | API preprocessing | Missing features are filled, unknown features are rejected |
| TC-10 | Drift calculation | Binary flags are excluded and score matches formula |

## 3. Test Report

Execution date: 2026-04-20

Result:

| Total | Passed | Failed |
|---|---:|---:|
| 11 | 11 | 0 |

Notes:
- Pytest completed successfully.
- A single MLflow filesystem backend deprecation warning was emitted, but it did not affect test success.

Acceptance decision:
- The software meets the acceptance criteria.
