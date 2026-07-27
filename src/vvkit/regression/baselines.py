"""Regression baseline store and tolerance drift detection."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaselineComparison:
    is_drifted: bool
    diffs: dict[str, float]


class BaselineStore:
    def __init__(self, baselines_dir: Path) -> None:
        self.baselines_dir = baselines_dir
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def get_baseline_path(self, study_name: str) -> Path:
        return self.baselines_dir / f"{study_name}.json"

    def save_baseline(self, study_name: str, metrics: dict[str, float]) -> None:
        path = self.get_baseline_path(study_name)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def compare(
        self, study_name: str, current_metrics: dict[str, float], rtol: float = 1e-3
    ) -> BaselineComparison:
        path = self.get_baseline_path(study_name)
        if not path.exists():
            return BaselineComparison(is_drifted=False, diffs={})

        baseline = json.loads(path.read_text(encoding="utf-8"))
        diffs = {}
        is_drifted = False

        for k, curr_v in current_metrics.items():
            if k in baseline:
                base_v = baseline[k]
                rel_diff = abs(curr_v - base_v) / max(abs(base_v), 1e-12)
                diffs[k] = rel_diff
                if rel_diff > rtol:
                    is_drifted = True

        return BaselineComparison(is_drifted=is_drifted, diffs=diffs)
