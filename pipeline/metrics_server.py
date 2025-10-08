"""FastAPI Prometheus metrics endpoint for drift monitoring."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from fastapi import FastAPI, Response
import prometheus_client
from prometheus_client import CollectorRegistry, Gauge
import uvicorn

try:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None

    class MlflowException(Exception):
        """Fallback MlflowException when mlflow is unavailable."""

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    drift_count: float
    drift_share: float


def _get_mlflow_client(tracking_uri: Optional[str]) -> MlflowClient:
    if MlflowClient is None:
        raise RuntimeError("mlflow is required to fetch drift metrics but is not installed.")
    kwargs = {}
    if tracking_uri:
        kwargs["tracking_uri"] = tracking_uri
    return MlflowClient(**kwargs)


def _fetch_drift_metrics(run_id: str, tracking_uri: Optional[str]) -> DriftMetrics:
    client = _get_mlflow_client(tracking_uri)
    try:
        run = client.get_run(run_id)
    except MlflowException as exc:
        logger.warning("Failed to fetch drift metrics for MLflow run %s: %s", run_id, exc)
        return DriftMetrics(drift_count=0.0, drift_share=0.0)

    metrics = run.data.metrics or {}
    return DriftMetrics(
        drift_count=float(metrics.get("drift_count", 0.0)),
        drift_share=float(metrics.get("drift_share", 0.0)),
    )


def create_metrics_app(run_id: str, *, tracking_uri: Optional[str] = None) -> Tuple[FastAPI, CollectorRegistry]:
    """Create a FastAPI app exposing the Evidently drift metrics from MLflow."""
    registry = CollectorRegistry()
    drifted_count_gauge = Gauge(
        "evidently_drifted_columns_count",
        "Number of drifted columns",
        registry=registry,
    )
    drifted_share_gauge = Gauge(
        "evidently_share_drifted_features",
        "Share of drifted features",
        registry=registry,
    )

    app = FastAPI()

    @app.get("/metrics")
    def get_metrics() -> Response:
        metrics = _fetch_drift_metrics(run_id, tracking_uri)
        drifted_count_gauge.set(metrics.drift_count)
        drifted_share_gauge.set(metrics.drift_share)
        payload = prometheus_client.generate_latest(registry)
        return Response(payload, media_type="text/plain")

    return app, registry


def run_metrics_server(
    run_id: str,
    *,
    tracking_uri: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
) -> None:
    """Run the metrics server via uvicorn."""
    if mlflow is None:
        raise RuntimeError("mlflow is required to run the drift metrics server but is not installed.")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    app, _ = create_metrics_app(run_id, tracking_uri=tracking_uri)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
