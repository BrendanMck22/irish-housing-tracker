"""Command-line entrypoint for the Irish house price gap pipeline."""
from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

from pipeline.data_processing import fetch_cso_dataset, prepare_price_gap_dataset
from pipeline.evidently_report import DriftReportResult, build_drift_report
from pipeline.metrics_server import run_metrics_server
from pipeline.mlflow_logging import log_training_run
from pipeline.model_training import TrainingResult, train_random_forest

try:
    from mlflow.exceptions import MlflowException
except Exception:
    MlflowException = Exception


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Irish house model pipeline.")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Override the MLflow tracking URI (defaults to env or http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip logging results to MLflow",
    )
    # parser.add_argument(
    #     "--drift-report-path",
    #     default="housing_drift_report.html",
    #     help="Path to store the Evidently drift report HTML",
    # )
    # parser.add_argument(
    #     "--baseline-end-year",
    #     type=int,
    #     default=2022,
    #     help="Final year to include in the drift baseline dataset",
    # )
    # parser.add_argument(
    #     "--current-year",
    #     type=int,
    #     default=2024,
    #     help="Year to treat as the current production slice for drift checks",
    # )
    # 2024 CSO data is incomplete so default to 2023 for now
    parser.add_argument(
        "--serve-metrics",
        action="store_true",
        help="Start the Prometheus metrics server after running the pipeline",
    )
    parser.add_argument(
        "--metrics-host",
        default="0.0.0.0",
        help="Host interface for the metrics server",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8000,
        help="Port for the metrics server",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> Tuple[TrainingResult, Optional[str], Optional[DriftReportResult]]:
    logger = logging.getLogger(__name__)

    logger.info("Fetching CSO dataset")
    raw_df = fetch_cso_dataset()

    logger.info("Preparing features and target variables")
    dataset = prepare_price_gap_dataset(raw_df)
    # Create a mask for training years
    train_mask = (dataset.prices['year'] >= 2010) & (dataset.prices['year'] <= 2016)

    # Slice all components consistently
    X_train = dataset.features.loc[train_mask]
    y_train = dataset.target.loc[train_mask]
    counties_train = dataset.counties.loc[train_mask]    
    logger.info("Training random forest model")
    training_result = train_random_forest(X_train, y_train, counties_train)

    logger.info("Running Evidently drift report")
    baseline_df = dataset.prices[(dataset.prices['year'] >= 2010) & (dataset.prices['year'] <= 2016)]
    current_df = dataset.prices[(dataset.prices['year'] >= 2017) & (dataset.prices['year'] <= 2023)]
    # send to airflow 
    data_dir = os.path.expanduser("~/airflow/data")
    os.makedirs(data_dir, exist_ok=True)

    # Define full file paths
    reference_path = os.path.join(data_dir, "housing_ref.csv")
    current_path = os.path.join(data_dir, "housing_curr.csv")

    # Export directly into Airflow’s data folder
    baseline_df.to_csv(reference_path, index=False)
    current_df.to_csv(current_path, index=False)


    drift_result: Optional[DriftReportResult]
    if baseline_df.empty or current_df.empty:
        logger.warning("Insufficient data to compute drift report (baseline or current slice empty)")
        drift_result = None
    else:
        drift_result = build_drift_report(
            baseline_df,
            current_df,
            output_path="housing_drift_report_old.html",
        )
        logger.info(
            "Drift summary: %s drifted columns (share=%.3f)",
            drift_result.drift_count,
            drift_result.drift_share,
        )
    run_id: Optional[str] = None
    if args.skip_mlflow:
        logger.info("Skipping MLflow logging as requested")
    else:
        logger.info("Logging results to MLflow")
        try:
            run_id = log_training_run(
                training_result,
                dataset.prices,
                tracking_uri="",
                artifact_dir=Path("artifacts"),
                drift_report_path=drift_result.html_path if drift_result else None,
                     drift_metrics={
                    "drift_count": float(drift_result.drift_count),
                    "drift_share": float(drift_result.drift_share),
                }
                if drift_result
                else None,
            )
        except MlflowException as exc:
            logger.error("MLflow logging failed: %s", exc)
            logger.info("Continuing without MLflow artefacts. Run with --skip-mlflow to hide this message.")
        except Exception:
            logger.exception("Unexpected MLflow failure; continuing without logging.")
        else:
            logger.info("Logged MLflow run %s", run_id)
        

    if args.serve_metrics:
        if run_id is None:
            logger.warning("Skipping metrics server because MLflow run ID is unavailable")
        else:
            logger.info(
                "Starting metrics server on %s:%d (MLflow run=%s)",
                args.metrics_host,
                args.metrics_port,
                run_id,
            )
            run_metrics_server(
                run_id=run_id,
                host=args.metrics_host,
                port=args.metrics_port,
                log_level=args.log_level.lower(),
            )

    return training_result, run_id, drift_result


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
