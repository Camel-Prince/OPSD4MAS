"""Initial-validation quality gate used before resumed PPO training."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def evaluate_initial_validation_gate(
    metrics: Mapping[str, Any],
    baselines: Mapping[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    """Compare validation metrics with ``baseline - tolerance`` thresholds."""
    tolerance = float(tolerance)
    if tolerance < 0:
        raise ValueError("initial validation gate tolerance must be non-negative")

    comparisons: Dict[str, Dict[str, Any]] = {}
    passed = True
    for name, baseline_value in baselines.items():
        baseline = float(baseline_value)
        minimum = baseline - tolerance
        observed_value = metrics.get(name)
        observed = float(observed_value) if observed_value is not None else None
        metric_passed = observed is not None and observed + 1e-12 >= minimum
        comparisons[name] = {
            "observed": observed,
            "baseline": baseline,
            "minimum": minimum,
            "passed": metric_passed,
        }
        passed = passed and metric_passed

    return {"passed": passed, "tolerance": tolerance, "metrics": comparisons}
