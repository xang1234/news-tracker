"""Drift detection and monitoring for the news-tracker pipeline.

Detects embedding distribution shifts, theme fragmentation, sentiment
calibration drift, and cluster centroid instability.

Usage:
    from src.monitoring import DriftService, DriftConfig

    service = DriftService(database)
    report = await service.run_daily_check()
    if report.has_issues:
        print(report.overall_severity)
"""

from src.monitoring.config import DriftConfig
from src.monitoring.schemas import DriftReport, DriftResult, DriftSeverity
from src.monitoring.service import DriftService

__all__ = [
    "DriftConfig",
    "DriftReport",
    "DriftResult",
    "DriftService",
    "DriftSeverity",
]
