import pytest


@pytest.mark.convergence(case="burgers_1d")
def test_pytest_plugin_marker() -> None:
    assert True
