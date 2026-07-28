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
