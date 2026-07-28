"""HTML and JSON emitters for verification study reports."""

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jinja2


@dataclass
class NormResultSummary:
    norm_name: str
    observed_order: float
    expected_order: float
    std_err: float | None
    r_squared: float | None
    order_passed: bool
    gci_fine: float
    asymptotic_ratio: float | None
    is_asymptotic: bool
    convergence_state: str


@dataclass
class StudyResultSummary:
    name: str
    norms: list[NormResultSummary]
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
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }
        h1 { color: #38bdf8; font-weight: 600; border-bottom: 2px solid #334155; padding-bottom: 10px; }
        .card { background: #1e293b; padding: 25px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
        .badge-pass { background: #14532d; color: #4ade80; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }
        .badge-fail { background: #7f1d1d; color: #f87171; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.95em; }
        th, td { padding: 12px 15px; border-bottom: 1px solid #334155; text-align: center; }
        th { color: #94a3b8; font-weight: 600; background: #0f172a; border-radius: 4px; }
        td:first-child, th:first-child { text-align: left; }
        .plot-container { text-align: center; margin-top: 20px; }
        .plot-container img { max-width: 100%; border-radius: 8px; border: 1px solid #334155; }
        .alert-warning { background: #451a03; border-left: 4px solid #f59e0b; padding: 12px; margin-top: 15px; border-radius: 4px; }
        .alert-critical { background: #4c0519; border-left: 4px solid #e11d48; padding: 12px; margin-top: 15px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>vvkit Verification Report: {{ summary.name }}</h1>
    
    <div class="card">
        <h2>Metrics Dashboard</h2>
        <table>
            <tr>
                <th>Norm</th>
                <th>Observed Order</th>
                <th>Expected Order</th>
                <th>Std Err</th>
                <th>R²</th>
                <th>GCI (Fine)</th>
                <th>Asymptotic Ratio (R)</th>
                <th>Convergence State</th>
                <th>Verdict</th>
            </tr>
            {% for norm in summary.norms %}
            <tr>
                <td><strong>{{ norm.norm_name }}</strong></td>
                <td>{{ "%.3f"|format(norm.observed_order) }}</td>
                <td>{{ "%.3f"|format(norm.expected_order) }}</td>
                <td>
                    {% if norm.std_err is not none %}
                    {{ "%.2e"|format(norm.std_err) }}
                    {% else %}
                    N/A
                    {% endif %}
                </td>
                <td>
                    {% if norm.r_squared is not none %}
                    {{ "%.4f"|format(norm.r_squared) }}
                    {% else %}
                    N/A
                    {% endif %}
                </td>
                <td>{{ "%.2e"|format(norm.gci_fine) }}</td>
                <td>
                    {% if norm.asymptotic_ratio is not none %}
                    {{ "%.3f"|format(norm.asymptotic_ratio) }}
                    {% else %}
                    N/A
                    {% endif %}
                </td>
                <td>{{ norm.convergence_state }}</td>
                <td>
                    {% if norm.order_passed %}
                    <span class="badge-pass">PASSED</span>
                    {% else %}
                    <span class="badge-fail">FAILED</span>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>

        {% for norm in summary.norms %}
            {% if norm.convergence_state == 'oscillatory' %}
            <div class="alert-warning">
                <strong>Warning ({{ norm.norm_name }}):</strong> Convergence is oscillatory (R_c &lt; 0). GCI estimates may be unreliable.
            </div>
            {% elif norm.convergence_state == 'divergent' %}
            <div class="alert-critical">
                <strong>Critical ({{ norm.norm_name }}):</strong> Study is divergent (|R_c| > 1). The discretization has failed to converge.
            </div>
            {% endif %}
            
            {% if not norm.is_asymptotic and norm.asymptotic_ratio %}
            <div class="alert-warning">
                <strong>Warning ({{ norm.norm_name }}):</strong> Solutions are outside the asymptotic range (|R - 1| > 0.1). Order estimates are unreliable.
            </div>
            {% endif %}
        {% endfor %}
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
    
    testcases = ""
    failures = 0
    for norm in summary.norms:
        failure_element = ""
        if not norm.order_passed:
            failures += 1
            err_msg = f"Observed order {norm.observed_order:.2f} != {norm.expected_order:.2f}"
            failure_element = f'\n            <failure message="{err_msg}"/>'
            
        testcases += f"""
        <testcase name="{summary.name}_{norm.norm_name}" classname="vvkit.convergence">{failure_element}
        </testcase>"""

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="vvkit" tests="{len(summary.norms)}" failures="{failures}" errors="0">{testcases}
    </testsuite>
</testsuites>
"""
    output_path.write_text(xml_content, encoding="utf-8")
