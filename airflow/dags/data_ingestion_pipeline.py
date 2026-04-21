from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/project"))
RAW_SOURCE = PROJECT_ROOT / "data" / "loan_data.csv"
INGESTED_DIR = PROJECT_ROOT / "data" / "raw_ingested"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HASH_FILE = INGESTED_DIR / ".last_hash"  # hidden file to store last known hash


def validate_raw_source() -> None:
    if not RAW_SOURCE.exists():
        raise FileNotFoundError(f"Raw source file not found: {RAW_SOURCE}")

    # Compute current file hash
    current_hash = hashlib.md5(RAW_SOURCE.read_bytes()).hexdigest()

    # Compare against last run's hash
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    if HASH_FILE.exists() and HASH_FILE.read_text().strip() == current_hash:
        raise AirflowSkipException("loan.csv unchanged since last run — skipping pipeline.")

    # File is new or changed — persist the new hash for next run
    HASH_FILE.write_text(current_hash)


def ingest_snapshot() -> None:
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    ds_nodash = datetime.utcnow().strftime("%Y%m%d")
    target = INGESTED_DIR / f"loan_{ds_nodash}.csv"
    shutil.copyfile(RAW_SOURCE, target)


def validate_processed_artifacts() -> None:
    required = [
        PROCESSED_DIR / "X_processed.csv",
        PROCESSED_DIR / "y.csv",
        PROCESSED_DIR / "split" / "X_train_processed.csv",
        PROCESSED_DIR / "split" / "X_test_processed.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing processed artifacts: " + ", ".join(missing))


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="credit_risk_data_ingestion",
    default_args=default_args,
    description="Ingest raw loan data and run processing stage",
    start_date=datetime(2026, 4, 1),
    schedule="0 2 * * *",
    catchup=False,
    tags=["credit-risk", "ingestion", "data"],
) as dag:

    validate_source = PythonOperator(
        task_id="validate_raw_source",
        python_callable=validate_raw_source,
    )

    ingest_data = PythonOperator(
        task_id="ingest_daily_snapshot",
        python_callable=ingest_snapshot,
    )

    process_data = BashOperator(
        task_id="process_data_stage",
        bash_command=(
            "cd {{ params.project_root }} && "
            "python src/data/process.py"
        ),
        params={"project_root": str(PROJECT_ROOT)},
    )

    validate_outputs = PythonOperator(
        task_id="validate_processed_outputs",
        python_callable=validate_processed_artifacts,
    )

    validate_source >> ingest_data >> process_data >> validate_outputs