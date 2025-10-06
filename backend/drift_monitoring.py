"""Deprecated entrypoint maintained for backwards compatibility.

Prefer using backend.metrics_server instead."""
from __future__ import annotations

from .metrics_server import DriftMetrics, create_metrics_app, run_metrics_server

__all__ = ["DriftMetrics", "create_metrics_app", "run_metrics_server"]
