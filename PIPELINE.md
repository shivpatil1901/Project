# Credit Risk MLOps Pipeline

This repository contains an end-to-end credit risk MLOps workflow for data processing, feature engineering, feature selection, model training, experiment tracking, API serving, UI inference, monitoring, and scheduled ingestion.

## 1. Repository Overview

### Core code paths

- [src/data/process.py](src/data/process.py): raw data cleaning, target creation, preprocessing, and train/test split artifacts.
- [src/features/engineer.py](src/features/engineer.py): correlation analysis, WoE/IV summaries, and engineered feature generation.
- [src/features/selection.py](src/features/selection.py): leakage removal, scaling, random-forest feature selection, and selected matrix export.
- [src/model/train.py](src/model/train.py): multi-model training for the local pipeline.
- [src/model/evaluate.py](src/model/evaluate.py): model evaluation and metrics export.
- [src/model/train_mlflow.py](src/model/train_mlflow.py): single-model training with MLflow logging.
- [src/api/main.py](src/api/main.py): FastAPI inference service, SHAP explainability, and Prometheus metrics.
- [streamlit_app.py](streamlit_app.py): Streamlit UI for interactive inference.
- [src/pipeline.py](src/pipeline.py): Python entry point for orchestrating the local pipeline.

### Supporting project assets

- [dvc.yaml](dvc.yaml): reproducible pipeline definition.
- [params.yaml](params.yaml): data paths, thresholds, model settings, and experiment parameters.
- [docker-compose.yml](docker-compose.yml): runtime stack for backend, frontend, MLflow, Prometheus, Grafana, and Airflow.
- [airflow/dags/data_ingestion_pipeline.py](airflow/dags/data_ingestion_pipeline.py): Airflow ingestion workflow.
- [monitoring/prometheus/prometheus.yml](monitoring/prometheus/prometheus.yml): Prometheus scrape configuration.
- [monitoring/grafana/dashboards/credit_risk_monitoring.json](monitoring/grafana/dashboards/credit_risk_monitoring.json): dashboard definition.
- [docs/HLD.md](docs/HLD.md), [docs/LLD.md](docs/LLD.md), [docs/testing.md](docs/testing.md): design and validation docs.

## 2. Data Flow

The pipeline currently uses [data/loan_data.csv](data/loan_data.csv) as the active raw input. The DVC stages transform that file into processed, engineered, selected, and evaluation artifacts.

### Stage 1: Data Processing

- Input: `data/loan_data.csv`
- Script: [src/data/process.py](src/data/process.py)
- Main tasks:
  - load and clean raw records
  - create the target label
  - remove high-missingness columns
  - convert categorical and date fields
  - generate numeric features
  - write train/test split artifacts

### Stage 2: Feature Engineering

- Input: processed feature matrix and target
- Script: [src/features/engineer.py](src/features/engineer.py)
- Main tasks:
  - compute correlation statistics
  - compute WoE/IV summaries
  - remove highly correlated features
  - create derived ratios and other engineered fields

### Stage 3: Feature Selection

- Input: engineered data
- Script: [src/features/selection.py](src/features/selection.py)
- Main tasks:
  - remove leakage-prone fields before selection
  - split and scale data
  - train a random forest selector
  - persist selected feature matrices and selector artifacts

### Stage 4: Model Training

- Input: selected training data
- Script: [src/model/train.py](src/model/train.py)
- Main tasks:
  - train multiple candidate models
  - persist trained model artifacts
  - maintain a model registry file

### Stage 5: Model Evaluation

- Input: selected test data and trained models
- Script: [src/model/evaluate.py](src/model/evaluate.py)
- Main tasks:
  - compute classification metrics
  - write summary metrics and detailed reports

## 3. Reproducible Pipeline With DVC

The pipeline is defined in [dvc.yaml](dvc.yaml) and can be reproduced with:

```bash
dvc repro
```

Tracked stages:

- `process_data`
- `feature_engineering`
- `feature_selection`
- `train_models`
- `evaluate_models`
- `train_model_mlflow`

Each stage declares its dependencies, parameters, and outputs so DVC reruns only the impacted steps when inputs change.

## 4. MLflow Experiment Tracking

The repository includes a separate MLflow training path through [src/model/train_mlflow.py](src/model/train_mlflow.py).

Typical usage:

```bash
python src/model/train_mlflow.py --model-name neural_network
```

The corresponding DVC stage is `train_model_mlflow`, which logs:

- model parameters
- metrics
- model artifact
- run metadata in [models/experiments/run_info.json](models/experiments/run_info.json)

## 5. Serving and Monitoring

### FastAPI backend

- Start with `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- Exposes endpoints for health, prediction, batch prediction, explanations, and metrics

### Streamlit frontend

- Start with `streamlit run streamlit_app.py`
- Calls the backend over REST using `BACKEND_URL`

### Prometheus and Grafana

- Prometheus scrapes metrics from the backend at `/metrics`
- Grafana dashboards visualize request counts, latency, and drift metrics

The monitoring stack is configured in [docker-compose.yml](docker-compose.yml) and [monitoring/prometheus/prometheus.yml](monitoring/prometheus/prometheus.yml).

## 6. Airflow Integration

Airflow is used for scheduled ingestion and can run independently of the frontend.

Relevant files:

- [airflow/dags/data_ingestion_pipeline.py](airflow/dags/data_ingestion_pipeline.py)
- [airflow/Dockerfile](airflow/Dockerfile)
- [airflow/config](airflow/config)
- [airflow/logs](airflow/logs)
- [airflow/plugins](airflow/plugins)

To start the Airflow services later, use the Airflow service set in [docker-compose.yml](docker-compose.yml).

## 7. Key Artifacts

- Processed data: [data/processed](data/processed)
- Engineered features: [data/features](data/features)
- Selected features: [data/features/selected](data/features/selected)
- Trained models: [models](models)
- Evaluation metrics: [models/evaluation/model_performance.csv](models/evaluation/model_performance.csv)
- MLflow run metadata: [models/experiments/run_info.json](models/experiments/run_info.json)

## 8. How To Run

### Local pipeline

```bash
dvc repro
```

### Individual stages

```bash
python src/data/process.py
python src/features/engineer.py
python src/features/selection.py
python src/model/train.py
python src/model/evaluate.py
```

### Full platform stack

```bash
docker compose up -d
```

### Frontend without Airflow

```bash
docker compose up -d frontend
```

### Airflow later

```bash
docker compose up -d airflow-postgres airflow-redis airflow-init airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer
```

## 9. Configuration Notes

- All core paths and thresholds are centralized in [params.yaml](params.yaml).
- The active raw input is configured as `data/loan_data.csv`.
- Leakage-prone features are removed before selection.
- The current runtime stack is split so the frontend can run independently from Airflow.

## 10. Validation

The repository includes unit tests in [tests](tests) and the current baseline report in [docs/testing.md](docs/testing.md).

Run tests with:

```bash
python -m pytest -q
```
