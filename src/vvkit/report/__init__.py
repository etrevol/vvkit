"""Report module initialization."""

from vvkit.report.emitters import (
    StudyResultSummary,
    emit_html_report,
    emit_json_report,
    emit_junit_xml,
)
from vvkit.report.plots import generate_convergence_plot

__all__ = [
    "StudyResultSummary",
    "emit_json_report",
    "emit_html_report",
    "emit_junit_xml",
    "generate_convergence_plot",
]
