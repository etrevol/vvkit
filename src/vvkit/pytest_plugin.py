"""pytest-vvkit plugin providing @pytest.mark.convergence marker support."""

from typing import Any


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "convergence(case): mark test to run a vvkit convergence verification study",
    )
