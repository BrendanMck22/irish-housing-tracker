"""
Airflow DAG: housing_drift_monitor
----------------------------------
Monitors housing dataset drift (via Prometheus + Evidently)
and retrains ML model if drift exceeds threshold.
Logs drift reports and retraining actions to the existing MLflow run.
"""

from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
import requests
import mlflow
import pandas as pd
import time
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from evidently import Report
from evidently.metrics import *
from evidently.presets import *

# Make sure we can import your preprocessing code
sys.path.append(os.path.expanduser("~/airflow/dags"))
from preprocess_data import fetch_and_preprocess_data

# --- CONFIG ---
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "irish_housing_price_gap"
DRIFT_THRESHOLD = 0.5
DATA_DIR = os.path.expanduser("~/airflow/data")

REFERENCE_FILE = os.path.join(DATA_DIR, "housing_ref.csv")    # 2010–2016
CURRENT_FILE = os.path.join(DATA_DIR, "housing_curr.csv")     # 2017 - 2023
LATEST_FILE = os.path.join(DATA_DIR, "latest_housing_curr.csv")  # new/latest dataset
REPORT_PATH = os.path.join(DATA_DIR, "housing_drift_report_new.html")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_latest_run_id():
    """Load the most recent MLflow run ID saved by the training script."""
    run_id_path = os.path.join(DATA_DIR, "current_run_id.txt")
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
            print(f"Found existing MLflow run ID: {run_id}")
            return run_id
    print("Warning: No saved run_id found. Will create a new MLflow run.")
    return None


# --- TASKS ---

@task
def fetch_latest_data():
    """Fetch and preprocess latest CSO housing data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df_latest = fetch_and_preprocess_data()
    df_latest.to_csv(LATEST_FILE, index=False)
    print(f"Latest housing data saved to {LATEST_FILE}")
    return LATEST_FILE


@task
def check_drift() -> float:
    """Fetch current drift value from Prometheus."""
    try:
        resp = requests.get(PROMETHEUS_URL, params={"query": "evidently_share_drifted_features"})
        data = resp.json().get("data", {}).get("result", [])
        if not data:
            print("Warning: No drift metric found.")
            return 0.0
        drift_value = float(data[0]["value"][1])
        print(f"Current drift share: {drift_value:.3f}")
        return drift_value
    except Exception as e:
        print(f"Error fetching drift: {e}")
        return 0.0


@task
def generate_drift_report():
    """Generate Evidently HTML drift report and log as MLflow artifact."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        ref_df = pd.read_csv(REFERENCE_FILE)
        curr_df = pd.read_csv(CURRENT_FILE)
        features_to_monitor_ref = [col for col in ref_df.columns if col != "year"]
        features_to_monitor_curr = [col for col in curr_df.columns if col != "year"]
        # Generate drift report
        report = Report(metrics=[DataDriftPreset()])
        report_eval = report.run(reference_data=ref_df[features_to_monitor_ref], current_data=ref_df[features_to_monitor_ref])
        report_eval.save_html(REPORT_PATH)
        # print(f"Drift report saved at {REPORT_PATH}")
        report_dict = report_eval.dict()
        # --- Extract drift summary ---
        drift_summary = next(
            (m for m in report_dict["metrics"] if m["metric_id"].startswith("DriftedColumnsCount")),
            None
        )

        if drift_summary:
            drift_count = drift_summary["value"]["count"]
            drift_share = drift_summary["value"]["share"]
        else:
            drift_count = drift_share = 0
        # Log to MLflow
        run_id = get_latest_run_id()
        print(f"drift share is {drift_share} and drift count is {drift_count}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(REPORT_PATH, artifact_path="drift_reports")
                mlflow.log_metric("drift_count", drift_count)
                mlflow.log_metric("drift_share", drift_share)


        else:
            with mlflow.start_run(run_name="Drift_Report_Run"):
                mlflow.log_artifact(REPORT_PATH, artifact_path="drift_reports")

        print("Drift report logged to MLflow successfully.")
    except Exception as e:
        print(f"Error generating drift report: {e}")


@task
def retrain_model():
    """Retrain the ML model using the latest dataset and log to MLflow."""
    print("Starting model retraining...")
    df = pd.read_csv(LATEST_FILE)

    # Encode categorical features
    df_encoded = pd.get_dummies(df, columns=["county", "dwelling_status"], drop_first=True)
    y = df_encoded["gap_ratio"]
    X = df_encoded.drop(columns=["mean_sale_price", "median_price", "gap_ratio"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Retrained model performance: MAE={mae:.4f}, R2={r2:.4f}")

    # Log retraining results
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="Retrained_Model"):
        mlflow.log_metric("MAE_retrain", mae)
        mlflow.log_metric("R2_retrain", r2)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="HousingPriceGapPredictor")

    print("New model retrained and logged to MLflow successfully.")


@task
def decide_and_retrain(drift_value: float):
    """Decide whether to retrain based on drift threshold."""
    print(f"Drift value received: {drift_value}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_id = get_latest_run_id()
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("drift_share", drift_value)
            if drift_value > DRIFT_THRESHOLD:
                print("Drift above threshold — retraining.")
                retrain_model()
                mlflow.log_param("triggered_by", f"drift_share>{DRIFT_THRESHOLD}")
            else:
                print("Drift stable — no retraining required.")
                mlflow.log_param("triggered_by", "stable_drift")
    else:
        with mlflow.start_run(run_name="DriftCheck_Run"):
            mlflow.log_metric("drift_share", drift_value)
            if drift_value > DRIFT_THRESHOLD:
                retrain_model()
                mlflow.log_param("triggered_by", f"drift_share>{DRIFT_THRESHOLD}")


# --- DAG CONFIG ---
default_args = {
    "owner": "brendan",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="housing_drift_monitor",
    default_args=default_args,
    description="Monitor housing data drift, generate report, and retrain if needed",
    schedule=timedelta(hours=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "housing", "drift"],
) as dag:
    latest_data = fetch_latest_data()
    drift_value = check_drift()
    generate_drift_report()
    decide_and_retrain(drift_value)
