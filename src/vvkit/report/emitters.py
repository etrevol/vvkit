"""HTML and JSON emitters for verification study reports."""

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jinja2


@dataclass
class StudyResultSummary:
    name: str
    observed_order: float
    expected_order: float
    order_passed: bool
    gci_fine: float
    asymptotic_ratio: float | None
    is_asymptotic: bool
    convergence_state: str
    plot_image_path: Path | None = None


def emit_json_report(summary: StudyResultSummary, output_path: Path) -> None:
    """Emit machine-readable JSON report contract."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_dict = asdict(summary)
    if summary_dict.get("plot_image_path"):
        summary_dict["plot_image_path"] = str(summary_dict["plot_image_path"])

    data = {
        "schema_version": 1,
        "summary": summary_dict,
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def emit_html_report(summary: StudyResultSummary, output_path: Path) -> None:
    """Emit self-contained offline HTML report using Jinja2 with embedded plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_b64 = ""
    if summary.plot_image_path and summary.plot_image_path.exists():
        plot_bytes = summary.plot_image_path.read_bytes()
        plot_b64 = base64.b64encode(plot_bytes).decode("utf-8")

    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>vvkit Report — {{ summary.name }}</title>
    <style>
        body { font-family: sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }
        h1 { color: #38bdf8; }
        .card { background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .badge-pass { background: #166534; color: #4ade80; padding: 4px; border-radius: 4px; }
        .badge-fail { background: #991b1b; color: #f87171; padding: 4px; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 8px 12px; border-bottom: 1px solid #334155; text-align: left; }
        th { color: #94a3b8; }
        .plot-container { text-align: center; margin-top: 20px; }
        .plot-container img { max-width: 100%; border-radius: 8px; border: 1px solid #334155; }
    </style>
</head>
<body>
    <h1>vvkit Verification Report: {{ summary.name }}</h1>
    <div class="card">
        <h2>Summary Verdict:
            {% if summary.order_passed %}
            <span class="badge-pass">PASSED</span>
            {% else %}
            <span class="badge-fail">FAILED</span>
            {% endif %}
        </h2>
        <table>
            <tr><th>Observed Order</th><td>{{ "%.3f"|format(summary.observed_order) }}</td></tr>
            <tr><th>Expected Order</th><td>{{ "%.3f"|format(summary.expected_order) }}</td></tr>
            <tr><th>GCI Fine Grid</th><td>{{ "%.2e"|format(summary.gci_fine) }}</td></tr>
            <tr><th>Convergence State</th><td>{{ summary.convergence_state }}</td></tr>
            <tr>
                <th>Asymptotic Ratio (R)</th>
                <td>
                    {% if summary.asymptotic_ratio %}
                    {{ "%.3f"|format(summary.asymptotic_ratio) }}
                    {% else %}
                    N/A
                    {% endif %}
                </td>
            </tr>
        </table>

        {% if summary.convergence_state == 'oscillatory' %}
        <div style="background: #451a03; border-left: 4px solid #f59e0b; padding: 10px; margin-top: 20px;">
            <strong>Warning:</strong> Convergence is oscillatory (R_c < 0). GCI estimates may be unreliable.
        </div>
        {% elif summary.convergence_state == 'divergent' %}
        <div style="background: #4c0519; border-left: 4px solid #e11d48; padding: 10px; margin-top: 20px;">
            <strong>Critical:</strong> Study is divergent (|R_c| > 1). The discretization has failed to converge.
        </div>
        {% endif %}
        
        {% if not summary.is_asymptotic and summary.asymptotic_ratio %}
        <div style="background: #451a03; border-left: 4px solid #f59e0b; padding: 10px; margin-top: 20px;">
            <strong>Warning:</strong> Solutions are outside the asymptotic range (|R - 1| > 0.1). Order estimates are unreliable.
        </div>
        {% endif %}
    </div>

    {% if plot_b64 %}
    <div class="card plot-container">
        <h3>Convergence Log-Log Plot</h3>
        <img src="data:image/png;base64,{{ plot_b64 }}" alt="Convergence Plot">
    </div>
    {% endif %}
</body>
</html>
"""
    rendered = jinja2.Template(html_template).render(summary=summary, plot_b64=plot_b64)
    output_path.write_text(rendered, encoding="utf-8")


def emit_junit_xml(summary: StudyResultSummary, output_path: Path) -> None:
    """Emit JUnit XML for CI integration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0 if summary.order_passed else 1
    failure_element = ""
    if not summary.order_passed:
        err_msg = f"Observed order {summary.observed_order:.2f} != {summary.expected_order:.2f}"
        failure_element = f'<failure message="{err_msg}"/>'

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="vvkit" tests="1" failures="{failures}" errors="0">
        <testcase name="{summary.name}" classname="vvkit.convergence">
            {failure_element}
        </testcase>
    </testsuite>
</testsuites>
"""
    output_path.write_text(xml_content, encoding="utf-8")
