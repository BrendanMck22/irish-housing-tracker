"""Evidently drift reporting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.metrics import *
from evidently.presets import *


@dataclass
class DriftReportResult:
    report: Report
    drift_count: int
    drift_share: float
    html_path: Path
    raw_report: dict


def build_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    output_path: Path | str = "housing_drift_report.html",
) -> DriftReportResult:
    """Run an Evidently data drift report and persist the HTML artefact."""
    report = Report([
    DataDriftPreset()
    ])
    output_report = report.run(reference_data=baseline_df, current_data=current_df)

    output_path = Path(output_path)
    output_report.save_html(str(output_path))
    report_dict = output_report.dict()

    # if hasattr(output_report, "as_dict"):
    #     report_dict = output_report.dict()
    # elif hasattr(report, "dict"):
    #     report_dict = output_report.dict()
    # else:
    #     raise AttributeError("Evidently report object does not expose as_dict/dict serialization")
    drift_summary = next(
        (
            metric
            for metric in report_dict.get("metrics", [])
            if metric.get("metric_id", "").startswith("DriftedColumnsCount")
        ),
        None,
    )

    if drift_summary:
        drift_count = int(drift_summary["value"].get("count", 0))
        drift_share = float(drift_summary["value"].get("share", 0.0))
    else:
        drift_count = 0
        drift_share = 0.0

    return DriftReportResult(
        report=report,
        drift_count=drift_count,
        drift_share=drift_share,
        html_path=output_path,
        raw_report=report_dict,
    )
