from pathlib import Path

from typer.testing import CliRunner

from vvkit.cli.main import app
from vvkit.config import MMSConfig, RefinementConfig, SolverConfig, StudyConfig, VVCaseConfig
from vvkit.report import StudyResultSummary, emit_html_report, emit_json_report, emit_junit_xml

runner = CliRunner()


def test_cli_init_and_run(tmp_path: Path) -> None:
    yaml_path = tmp_path / "vvcase.yaml"
    result_init = runner.invoke(app, ["init", "--path", str(yaml_path)])
    assert result_init.exit_code == 0
    assert yaml_path.exists()

    result_run = runner.invoke(app, ["run", "--config-path", str(yaml_path)])
    assert result_run.exit_code == 0


def test_report_emitters(tmp_path: Path) -> None:
    summary = StudyResultSummary(
        name="test_study",
        observed_order=2.01,
        expected_order=2.0,
        order_passed=True,
        gci_fine=0.0012,
        asymptotic_ratio=0.99,
        is_asymptotic=True,
        convergence_state="monotonic",
    )

    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    junit_path = tmp_path / "report.xml"

    emit_json_report(summary, json_path)
    emit_html_report(summary, html_path)
    emit_junit_xml(summary, junit_path)

    assert json_path.exists()
    assert html_path.exists()
    assert junit_path.exists()
    assert "PASSED" in html_path.read_text(encoding="utf-8")


def test_config_models() -> None:
    cfg = VVCaseConfig(
        name="test",
        solver=SolverConfig(type="command", command=["./run"]),
        mms=MMSConfig(operator="u_x", solution="x^2"),
        study=StudyConfig(refinement=RefinementConfig(parameter="n", values=[10, 20])),
    )
    assert cfg.name == "test"
