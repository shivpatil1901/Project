# Credit Risk MLOps Project

End-to-end MLOps workflow for credit default prediction with data pipeline orchestration, experiment tracking, model serving, UI inference, explainability, monitoring, and Airflow scheduling.

## Repository Contents

- Data processing, feature engineering, feature selection, training, and evaluation pipeline in [src](src)
- DVC pipeline orchestration in [dvc.yaml](dvc.yaml)
- MLflow experiment training entrypoint in [src/model/train_mlflow.py](src/model/train_mlflow.py)
- FastAPI inference service in [src/api/main.py](src/api/main.py)
- Streamlit frontend in [streamlit_app.py](streamlit_app.py)
- Merged platform stack (API, UI, MLflow, Prometheus, Grafana, Airflow) in [docker-compose.yml](docker-compose.yml)
- Airflow ingestion DAG in [airflow/dags/data_ingestion_pipeline.py](airflow/dags/data_ingestion_pipeline.py)
- Unit tests in [tests](tests)
- Design and testing docs in [docs](docs)

## Architecture Overview

Core blocks are loosely coupled:

- Frontend and backend communicate only via REST API
- Backend loads model artifacts and feature metadata
- Pipeline stages are reproducible through DVC
- Airflow schedules ingestion workflow
- Prometheus scrapes metrics from backend
- Grafana visualizes operational and business metrics

See:
- High-level design: [docs/HLD.md](docs/HLD.md)
- Low-level design: [docs/LLD.md](docs/LLD.md)
- Test plan and report: [docs/testing.md](docs/testing.md)

## Repository Structure

- [src/data/process.py](src/data/process.py): Data cleaning and preprocessing
- [src/features/engineer.py](src/features/engineer.py): Feature engineering
- [src/features/selection.py](src/features/selection.py): Leakage handling, scaling, RF feature selection
- [src/model/train.py](src/model/train.py): Multi-model training
- [src/model/evaluate.py](src/model/evaluate.py): Evaluation and metrics export
- [src/model/train_mlflow.py](src/model/train_mlflow.py): Single-model MLflow training
- [src/api/main.py](src/api/main.py): Inference API, SHAP explainability, Prometheus metrics
- [streamlit_app.py](streamlit_app.py): Interactive inference UI
- [params.yaml](params.yaml): Configurations and hyperparameters
- [requirements_pipeline.txt](requirements_pipeline.txt): Python dependencies

## Data Input Strategy

The pipeline now uses only an input file:

- Active raw input path is configured as data/loan_data.csv in [params.yaml](params.yaml)
- One-time sampled file is stored at [data/loan_data.csv](data/loan_data.csv)

## Prerequisites

- Python 3.10 recommended
- DVC installed in the same environment used to run the pipeline
- Docker Desktop for the merged platform stack

## Setup

1) Create or activate environment and install dependencies

    conda activate ml-gpu
    pip install -r requirements_pipeline.txt

2) Verify key tools

    python --version
    dvc --version

3) Ensure sampled input exists

    dir data\loan_data.csv

## Run the Pipeline

### Option A: DVC orchestration

    dvc repro

### Option B: Python orchestration script

    python run_pipeline.py

### Option C: Individual stages

    python src/data/process.py
    python src/features/engineer.py
    python src/features/selection.py
    python src/model/train.py
    python src/model/evaluate.py

## MLflow Training

Train one configured model and log metrics/artifacts:

    python src/model/train_mlflow.py --model-name neural_network

Change model and hyperparameters through [params.yaml](params.yaml) or DVC experiment overrides.

## Run Experiments with DVC + MLflow

Use `dvc exp run` to launch tracked experiments while changing model choice and hyperparameters. Experiments use the `train_model_mlflow` stage in [dvc.yaml](dvc.yaml), and each run is logged to MLflow.

### Run one model experiment

    dvc exp run train_model_mlflow -S experiment.model_name=random_forest

### Example commands for different models

    dvc exp run train_model_mlflow -S experiment.model_name=logistic_regression
    dvc exp run train_model_mlflow -S experiment.model_name=random_forest
    dvc exp run train_model_mlflow -S experiment.model_name=gradient_boosting
    dvc exp run train_model_mlflow -S experiment.model_name=neural_network

### Override hyperparameters per experiment

    dvc exp run train_model_mlflow -S experiment.model_name=random_forest -S models.random_forest.n_estimators=300
    dvc exp run train_model_mlflow -S experiment.model_name=gradient_boosting -S models.gradient_boosting.n_estimators=200
    dvc exp run train_model_mlflow -S experiment.model_name=neural_network -S models.neural_network.epochs=30 -S models.neural_network.learning_rate_init=0.0005

### Compare and apply experiments

List experiment results:

    dvc exp show

Apply the best experiment to workspace:

    dvc exp apply <exp_name>

### View MLflow runs

After running experiments, start MLflow UI and compare runs:

    mlflow ui

Run metadata and latest run id are also saved in [models/experiments/run_info.json](models/experiments/run_info.json).

## FastAPI Backend

Start backend:

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Swagger docs:
- http://127.0.0.1:8000/docs

Primary endpoints:
- GET /health
- POST /predict
- POST /predict-batch
- POST /explain
- GET /metrics

## Streamlit Frontend

Start UI:

    streamlit run streamlit_app.py

Default backend URL is configured via BACKEND_URL environment variable and points to http://127.0.0.1:8000 by default.

## Platform Services via Docker Compose

Start full stack from project root:

    docker compose up -d

Services:
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Airflow API/Web: http://localhost:8081

Note:
- In this merged compose file, Airflow API server maps host port `8081` to container port `8080`.
- `credit-risk-airflow-init` exits after setup; this is expected behavior.

Provisioned dashboard files:
- [monitoring/grafana/dashboards/credit_risk_monitoring.json](monitoring/grafana/dashboards/credit_risk_monitoring.json)
- [monitoring/grafana/provisioning/dashboards/dashboards.yml](monitoring/grafana/provisioning/dashboards/dashboards.yml)

Tracked metrics include:
- Request count
- API and business latency
- Drift score and drift distribution

## Airflow Files and Services (Merged Compose)

Airflow DAG file in this repo:
- [airflow/dags/data_ingestion_pipeline.py](airflow/dags/data_ingestion_pipeline.py)

Airflow runtime files used by `docker-compose.yml`:
- [airflow/Dockerfile](airflow/Dockerfile)
- [airflow/.env](airflow/.env)
- [airflow/config](airflow/config)
- [airflow/logs](airflow/logs)
- [airflow/plugins](airflow/plugins)

Airflow services started by merged compose:
- `airflow-postgres`
- `airflow-redis`
- `airflow-init`
- `airflow-apiserver`
- `airflow-scheduler`
- `airflow-dag-processor`
- `airflow-worker`
- `airflow-triggerer`

Typical commands from project root:

    docker compose exec --user airflow airflow-scheduler airflow dags list
    docker compose exec --user airflow airflow-scheduler airflow dags trigger credit_risk_data_ingestion
    docker compose exec --user airflow airflow-scheduler airflow dags list-runs credit_risk_data_ingestion

Note:
- Run Airflow CLI commands as user `airflow` in this setup.

## Unit Tests

Run tests:

    python -m pytest -q

Test suite location:
- [tests](tests)

Current baseline report is documented in [docs/testing.md](docs/testing.md).

## Key Outputs

- Processed data: [data/processed](data/processed)
- Engineered and selected features: [data/features](data/features)
- Trained models: [models](models)
- Evaluation metrics: [models/evaluation/model_performance.csv](models/evaluation/model_performance.csv)
- MLflow run info: [models/experiments/run_info.json](models/experiments/run_info.json)

## Troubleshooting

### DVC lock corruption

If DVC reports rwlock JSON corruption:

    Remove-Item .dvc\tmp\rwlock -Force
    Remove-Item .dvc\tmp\rwlock.lock -Force
    Remove-Item .dvc\tmp\state.lock -Force

Then rerun:

    dvc repro


## Related Documents

- Pipeline overview: [PIPELINE.md](PIPELINE.md)
- Setup notes: [SETUP_COMPLETE.md](SETUP_COMPLETE.md)
- HLD: [docs/HLD.md](docs/HLD.md)
- LLD: [docs/LLD.md](docs/LLD.md)
- Testing plan/report: [docs/testing.md](docs/testing.md)
- AI disclosure appendix: [docs/AI_DISCLOSURE_APPENDIX.md](docs/AI_DISCLOSURE_APPENDIX.md)
