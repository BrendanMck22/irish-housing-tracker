"""MLflow logging helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas

from .model_training import TrainingResult


def _reset_active_runs() -> None:
    while mlflow.active_run() is not None:
        mlflow.end_run()


def log_training_run(
    training_result: TrainingResult,
    prices: pd.DataFrame,
    *,
    run_name: str = "RF_Predict_Gap_2010_2023",
    experiment_name: str = "irish_housing_price_gap",
    tracking_uri: Optional[str] = None,
    artifact_dir: Optional[Path] = None,
) -> str:
    """Log model artefacts and metrics to MLflow."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    if uri:
        mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(experiment_name)
    _reset_active_runs()

    artifact_dir = artifact_dir or Path.cwd()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "county_year_metrics.csv"

    run_id = ""
    try:
        with mlflow.start_run(run_name=run_name) as active_run:
            run_id = active_run.info.run_id

            mlflow.log_param("target", "gap_ratio")
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", training_result.model.n_estimators)
            mlflow.log_param("train_years", "2010-2023")

            mlflow.log_metric("MAE_total", training_result.mae_total)
            mlflow.log_metric("R2_total", training_result.r2_total)

            for _, row in training_result.metrics_per_county_year.iterrows():
                county = row['county']
                year = int(row['year'])
                mlflow.log_metric(f"MAE_{county}", float(row['MAE']), step=year)
                mlflow.log_metric(f"R2_{county}", float(row['R2']), step=year)

            mean_gap_by_county = prices.groupby('county')['gap_ratio'].mean()
            mlflow.log_metrics({f"avg_gap_{county}": float(gap) for county, gap in mean_gap_by_county.items()})

            training_result.metrics_per_county_year.to_csv(metrics_path, index=False)
            dataset = from_pandas(prices, name="price_gap_dataset")
            mlflow.log_input(dataset, context="training")
            mlflow.log_artifact(str(metrics_path))

            mlflow.sklearn.log_model(
                training_result.model,
                artifact_path="HousingPriceGapPredictor",
                input_example=training_result.input_example,
                signature=training_result.signature,
            )
            os.makedirs(os.path.expanduser("~/airflow/data"), exist_ok=True)
            with open(os.path.expanduser("~/airflow/data/current_run_id.txt"), "w") as f:
                f.write(run_id)
    finally:
        if metrics_path.exists():
            metrics_path.unlink()

    return run_id
