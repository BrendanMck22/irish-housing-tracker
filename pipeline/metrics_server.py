"""FastAPI Prometheus metrics endpoint for drift monitoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fastapi import FastAPI, Response
import prometheus_client
from prometheus_client import CollectorRegistry, Gauge
import uvicorn


@dataclass
class DriftMetrics:
    drift_count: int
    drift_share: float


def create_metrics_app(metrics: DriftMetrics) -> Tuple[FastAPI, CollectorRegistry]:
    """Create a FastAPI app exposing the Evidently drift metrics."""
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
        drifted_count_gauge.set(metrics.drift_count)
        drifted_share_gauge.set(metrics.drift_share)
        payload = prometheus_client.generate_latest(registry)
        return Response(payload, media_type="text/plain")

    return app, registry


def run_metrics_server(
    metrics: DriftMetrics,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
) -> None:
    """Run the metrics server via uvicorn."""
    app, _ = create_metrics_app(metrics)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
