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

"""Norms module initialization."""

from vvkit.norms.norms import compute_l1_norm, compute_l2_norm, compute_linf_norm
from vvkit.norms.quadrature import cell_average_1d

__all__ = ["compute_l1_norm", "compute_l2_norm", "compute_linf_norm", "cell_average_1d"]
