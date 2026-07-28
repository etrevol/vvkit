# Copyright 2026 Artem Holovashchenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTML, JSON, and JUnit XML emitters for verification study reports."""

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jinja2


@dataclass
class GCITableRow:
    """A single row in the GCI detail table."""
    grid_triplet: str
    gci_fine: float
    safety_factor: float
    asymptotic_ratio: float | None
    is_asymptotic: bool
    convergence_condition_rc: float
    convergence_state: str


@dataclass
class EnvironmentInfo:
    """Provenance metadata for reproducibility."""
    vvkit_version: str
    platform: str
    timestamp: str
    python_version: str
    config_echo: str


@dataclass
class MMSDiagnostics:
    """MMS-specific diagnostic information."""
    operator_str: str
    solution_str: str
    vanished_terms: list[str]
    is_positive: bool


@dataclass
class ConservationResultSummary:
    """Summary of a single conservation check."""
    quantity: str
    field_name: str
    is_conserved: bool
    final_imbalance: float
    departure_step: int | None
    plot_image_path: Path | None = None


@dataclass
class NormResultSummary:
    """Per-norm convergence metrics."""
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
    pairwise_orders: list[float] = field(default_factory=list)
    errors_per_grid: list[float] = field(default_factory=list)


@dataclass
class StudyResultSummary:
    """Complete result contract for a verification study."""
    name: str
    norms: list[NormResultSummary]
    plot_image_path: Path | None = None
    grid_sizes: list[float] = field(default_factory=list)
    gci_table: list[GCITableRow] = field(default_factory=list)
    environment: EnvironmentInfo | None = None
    mms_diagnostics: MMSDiagnostics | None = None
    conservation_results: list[ConservationResultSummary] = field(default_factory=list)


def _serialize_summary(summary: StudyResultSummary) -> dict[str, Any]:
    """Convert StudyResultSummary to a JSON-safe dictionary."""
    d = asdict(summary)
    if d.get("plot_image_path"):
        d["plot_image_path"] = str(d["plot_image_path"])
    for cr in d.get("conservation_results", []):
        if cr.get("plot_image_path"):
            cr["plot_image_path"] = str(cr["plot_image_path"])
    return d


def emit_json_report(summary: StudyResultSummary, output_path: Path) -> None:
    """Emit machine-readable JSON report contract."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "summary": _serialize_summary(summary),
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def emit_html_report(summary: StudyResultSummary, output_path: Path) -> None:
    """Emit self-contained offline HTML report with embedded plots.

    The report includes: metrics dashboard, pairwise order table, error table,
    GCI table, convergence plot, asymptotic diagnostics, MMS diagnostics,
    conservation results, and environment provenance.
    Cites: PROJECT_SPEC.md Milestone M5.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_b64 = ""
    if summary.plot_image_path and summary.plot_image_path.exists():
        plot_b64 = base64.b64encode(summary.plot_image_path.read_bytes()).decode("utf-8")

    cons_plots_b64: dict[str, str] = {}
    for cr in summary.conservation_results:
        if cr.plot_image_path and cr.plot_image_path.exists():
            cons_plots_b64[cr.quantity] = base64.b64encode(
                cr.plot_image_path.read_bytes()
            ).decode("utf-8")

    all_passed = all(ns.order_passed for ns in summary.norms)

    max_pairwise = max((len(ns.pairwise_orders) for ns in summary.norms), default=0)

    html_template = _HTML_TEMPLATE
    rendered = jinja2.Template(html_template).render(
        summary=summary,
        plot_b64=plot_b64,
        cons_plots_b64=cons_plots_b64,
        all_passed=all_passed,
        max_pairwise=max_pairwise,
    )
    output_path.write_text(rendered, encoding="utf-8")


def emit_junit_xml(summary: StudyResultSummary, output_path: Path) -> None:
    """Emit JUnit XML for CI integration with actionable failure messages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    testcases = ""
    failures = 0
    for norm in summary.norms:
        failure_element = ""
        if not norm.order_passed:
            failures += 1
            tol = abs(norm.observed_order - norm.expected_order)
            asymp_str = (
                f"within asymptotic range (R={norm.asymptotic_ratio:.3f})"
                if norm.is_asymptotic
                else "outside asymptotic range"
            )
            err_msg = (
                f"Observed order {norm.observed_order:.2f} != expected "
                f"{norm.expected_order:.2f} (delta={tol:.2f}); "
                f"convergence is {norm.convergence_state}, {asymp_str}"
            )
            if norm.is_asymptotic and norm.convergence_state == "monotonic":
                err_msg += " — discretization is the likely cause"
            failure_element = f'\n            <failure message="{err_msg}"/>'

        testcases += f"""\n        <testcase name="{summary.name}_{norm.norm_name}" classname="vvkit.convergence">{failure_element}\n        </testcase>"""

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="vvkit" tests="{len(summary.norms)}" failures="{failures}" errors="0">{testcases}
    </testsuite>
</testsuites>
"""
    output_path.write_text(xml_content, encoding="utf-8")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>vvkit Report — {{ summary.name }}</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; line-height: 1.5; }
        h1 { color: #38bdf8; font-weight: 600; border-bottom: 2px solid #334155; padding-bottom: 10px; }
        h2 { color: #94a3b8; font-weight: 500; margin-top: 0; }
        .card { background: #1e293b; padding: 25px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }
        .badge-pass { background: #14532d; color: #4ade80; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }
        .badge-fail { background: #7f1d1d; color: #f87171; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }
        .badge-verdict { font-size: 1.1em; padding: 6px 16px; border-radius: 6px; margin-left: 15px; vertical-align: middle; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }
        th, td { padding: 10px 12px; border-bottom: 1px solid #334155; text-align: center; }
        th { color: #94a3b8; font-weight: 600; background: #0f172a; }
        td:first-child, th:first-child { text-align: left; }
        .plot-container { text-align: center; margin-top: 20px; }
        .plot-container img { max-width: 100%; border-radius: 8px; border: 1px solid #334155; }
        .alert-warning { background: #451a03; border-left: 4px solid #f59e0b; padding: 12px; margin-top: 15px; border-radius: 4px; }
        .alert-critical { background: #4c0519; border-left: 4px solid #e11d48; padding: 12px; margin-top: 15px; border-radius: 4px; }
        .alert-info { background: #0c2d48; border-left: 4px solid #38bdf8; padding: 12px; margin-top: 15px; border-radius: 4px; }
        .footer { margin-top: 30px; padding: 20px; background: #1e293b; border-radius: 12px; font-size: 0.85em; color: #64748b; }
        .footer strong { color: #94a3b8; }
        details { margin-top: 10px; }
        summary { cursor: pointer; color: #38bdf8; font-weight: 500; }
        details pre { background: #0f172a; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 0.85em; color: #cbd5e1; margin-top: 8px; }
        .mono { font-family: 'Consolas', 'Monaco', monospace; font-size: 0.9em; }
        .text-warn { color: #f59e0b; }
        .text-ok { color: #4ade80; }
        .text-err { color: #f87171; }
    </style>
</head>
<body>
    <h1>vvkit Verification Report: {{ summary.name }}
        {% if all_passed %}
        <span class="badge-pass badge-verdict">PASSED</span>
        {% else %}
        <span class="badge-fail badge-verdict">FAILED</span>
        {% endif %}
    </h1>

    {# ---- Section 1: Summary Metrics Dashboard ---- #}
    <div class="card">
        <h2>Convergence Metrics</h2>
        <table>
            <tr>
                <th>Norm</th>
                <th>Observed p</th>
                <th>Expected p</th>
                <th>Std Err</th>
                <th>R&sup2;</th>
                <th>GCI (Fine)</th>
                <th>R (Asymp.)</th>
                <th>Conv. State</th>
                <th>Verdict</th>
            </tr>
            {% for norm in summary.norms %}
            <tr>
                <td><strong>{{ norm.norm_name }}</strong></td>
                <td>{{ "%.4f"|format(norm.observed_order) }}</td>
                <td>{{ "%.1f"|format(norm.expected_order) }}</td>
                <td>{% if norm.std_err is not none %}{{ "%.2e"|format(norm.std_err) }}{% else %}N/A{% endif %}</td>
                <td>{% if norm.r_squared is not none %}{{ "%.5f"|format(norm.r_squared) }}{% else %}N/A{% endif %}</td>
                <td class="mono">{{ "%.3e"|format(norm.gci_fine) }}</td>
                <td>{% if norm.asymptotic_ratio is not none %}{{ "%.4f"|format(norm.asymptotic_ratio) }}{% else %}N/A{% endif %}</td>
                <td>{{ norm.convergence_state }}</td>
                <td>{% if norm.order_passed %}<span class="badge-pass">PASS</span>{% else %}<span class="badge-fail">FAIL</span>{% endif %}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    {# ---- Section 2: Pairwise Order Detail ---- #}
    {% if max_pairwise > 0 %}
    <div class="card">
        <h2>Pairwise Observed Orders</h2>
        <p style="color:#64748b;">Per-grid-pair orders reveal whether p is drifting (a sign coarse grids are outside the asymptotic range).</p>
        <table>
            <tr>
                <th>Grid Pair</th>
                {% for norm in summary.norms %}<th>{{ norm.norm_name }}</th>{% endfor %}
            </tr>
            {% for i in range(max_pairwise) %}
            <tr>
                <td class="mono">
                    {% if summary.grid_sizes|length > i + 1 %}
                    {{ "%.4g"|format(summary.grid_sizes[i]) }} &rarr; {{ "%.4g"|format(summary.grid_sizes[i+1]) }}
                    {% else %}
                    pair {{ i + 1 }}
                    {% endif %}
                </td>
                {% for norm in summary.norms %}
                <td>{% if i < norm.pairwise_orders|length %}{{ "%.4f"|format(norm.pairwise_orders[i]) }}{% else %}&mdash;{% endif %}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    {# ---- Section 3: Error Table ---- #}
    {% if summary.grid_sizes %}
    <div class="card">
        <h2>Error Norms per Grid Level</h2>
        <table>
            <tr>
                <th>Grid Size (h)</th>
                {% for norm in summary.norms %}<th>{{ norm.norm_name }}</th>{% endfor %}
            </tr>
            {% for i in range(summary.grid_sizes|length) %}
            <tr>
                <td class="mono">{{ "%.4e"|format(summary.grid_sizes[i]) }}</td>
                {% for norm in summary.norms %}
                <td class="mono">{% if i < norm.errors_per_grid|length %}{{ "%.4e"|format(norm.errors_per_grid[i]) }}{% else %}&mdash;{% endif %}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    {# ---- Section 4: GCI Table ---- #}
    {% if summary.gci_table %}
    <div class="card">
        <h2>Grid Convergence Index (GCI)</h2>
        <table>
            <tr>
                <th>Grid Triplet</th>
                <th>GCI (Fine)</th>
                <th>F<sub>s</sub></th>
                <th>R (Asymp.)</th>
                <th>R<sub>c</sub></th>
                <th>Conv. State</th>
                <th>In Asymp. Range</th>
            </tr>
            {% for row in summary.gci_table %}
            <tr>
                <td class="mono">{{ row.grid_triplet }}</td>
                <td class="mono">{{ "%.3e"|format(row.gci_fine) }}</td>
                <td>{{ "%.2f"|format(row.safety_factor) }}</td>
                <td>{% if row.asymptotic_ratio is not none %}{{ "%.4f"|format(row.asymptotic_ratio) }}{% else %}N/A{% endif %}</td>
                <td>{{ "%.4f"|format(row.convergence_condition_rc) }}</td>
                <td>{{ row.convergence_state }}</td>
                <td>{% if row.is_asymptotic %}<span class="text-ok">Yes</span>{% else %}<span class="text-warn">No</span>{% endif %}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    {# ---- Section 5: Convergence Plot ---- #}
    {% if plot_b64 %}
    <div class="card plot-container">
        <h2>Convergence Log–Log Plot</h2>
        <img src="data:image/png;base64,{{ plot_b64 }}" alt="Convergence Plot">
    </div>
    {% endif %}

    {# ---- Section 6: Asymptotic Diagnostics ---- #}
    {% for norm in summary.norms %}
        {% if norm.convergence_state == 'oscillatory' %}
        <div class="alert-warning">
            <strong>Warning ({{ norm.norm_name }}):</strong> Convergence is oscillatory (R<sub>c</sub> &lt; 0). GCI estimates may be unreliable.
        </div>
        {% elif norm.convergence_state == 'divergent' %}
        <div class="alert-critical">
            <strong>Critical ({{ norm.norm_name }}):</strong> Study is <strong>divergent</strong> (|R<sub>c</sub>| &gt; 1). The discretization has failed to converge. Order and GCI values are meaningless.
        </div>
        {% endif %}
        {% if not norm.is_asymptotic and norm.asymptotic_ratio is not none %}
        <div class="alert-warning">
            <strong>Warning ({{ norm.norm_name }}):</strong> Solutions are outside the asymptotic range (|R &minus; 1| &gt; 0.1, R={{ "%.4f"|format(norm.asymptotic_ratio) }}). Order estimates and GCI are not reliable.
        </div>
        {% endif %}
    {% endfor %}

    {# ---- Section 7: MMS Diagnostics ---- #}
    {% if summary.mms_diagnostics %}
    <div class="card">
        <h2>MMS Diagnostics</h2>
        <p><strong>Operator:</strong> <code>{{ summary.mms_diagnostics.operator_str }}</code></p>
        <p><strong>Manufactured Solution:</strong> <code>{{ summary.mms_diagnostics.solution_str }}</code></p>
        {% if summary.mms_diagnostics.vanished_terms %}
        <div class="alert-warning">
            <strong>Vanished Terms:</strong> The following operator terms are identically zero with the chosen solution and are therefore <em>not tested</em>:
            <ul>{% for t in summary.mms_diagnostics.vanished_terms %}<li><code>{{ t }}</code></li>{% endfor %}</ul>
        </div>
        {% else %}
        <div class="alert-info"><strong>All operator terms are exercised</strong> by the manufactured solution.</div>
        {% endif %}
        {% if not summary.mms_diagnostics.is_positive %}
        <div class="alert-warning">
            <strong>Positivity Warning:</strong> The manufactured solution is not strictly positive over the domain. Physical solvers requiring positive density/pressure/temperature may crash.
        </div>
        {% endif %}
    </div>
    {% endif %}

    {# ---- Section 8: Conservation Results ---- #}
    {% if summary.conservation_results %}
    <div class="card">
        <h2>Conservation Checks</h2>
        <table>
            <tr><th>Quantity</th><th>Field</th><th>Conserved</th><th>Final Imbalance</th><th>Departure Step</th></tr>
            {% for cr in summary.conservation_results %}
            <tr>
                <td><strong>{{ cr.quantity }}</strong></td>
                <td>{{ cr.field_name }}</td>
                <td>{% if cr.is_conserved %}<span class="text-ok">Yes</span>{% else %}<span class="text-err">No</span>{% endif %}</td>
                <td class="mono">{{ "%.3e"|format(cr.final_imbalance) }}</td>
                <td>{% if cr.departure_step is not none %}{{ cr.departure_step }}{% else %}&mdash;{% endif %}</td>
            </tr>
            {% endfor %}
        </table>
        {% for cr in summary.conservation_results %}
            {% if cr.quantity in cons_plots_b64 %}
            <div class="plot-container" style="margin-top:20px;">
                <h3>Imbalance Time Series: {{ cr.quantity }}</h3>
                <img src="data:image/png;base64,{{ cons_plots_b64[cr.quantity] }}" alt="Conservation Plot — {{ cr.quantity }}">
            </div>
            {% endif %}
        {% endfor %}
    </div>
    {% endif %}

    {# ---- Section 9: Environment Provenance ---- #}
    {% if summary.environment %}
    <div class="footer">
        <strong>vvkit</strong> {{ summary.environment.vvkit_version }} &middot;
        <strong>Platform:</strong> {{ summary.environment.platform }} &middot;
        <strong>Python:</strong> {{ summary.environment.python_version }} &middot;
        <strong>Generated:</strong> {{ summary.environment.timestamp }}
        <details>
            <summary>Full Configuration</summary>
            <pre>{{ summary.environment.config_echo }}</pre>
        </details>
    </div>
    {% endif %}
</body>
</html>
"""
