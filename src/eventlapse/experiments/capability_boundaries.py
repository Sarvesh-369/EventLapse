import json
from pathlib import Path
from typing import Dict, Any, List
from eventlapse.models.base import BaseVideoModel
from eventlapse.evaluation.exact_match import compute_wilson_score_interval
from eventlapse.evaluation.boundaries import estimate_operational_boundary

def run_capability_boundary_experiment(
    model: BaseVideoModel,
    task_name: str,
    dataset_manifest: List[Dict[str, Any]],
    tau: float = 0.80,
    harder_when: str = "increasing"
) -> Dict[str, Any]:
    # Group predictions by control_value
    by_param: Dict[float, List[bool]] = {}

    for item in dataset_manifest:
        val = float(item["control_value"])
        if val not in by_param:
            by_param[val] = []
        is_correct = item.get("exact_match_result", False)
        by_param[val].append(is_correct)

    accuracy_data = {}
    sweep_results = []

    for val in sorted(by_param.keys()):
        results = by_param[val]
        num_correct = sum(results)
        num_total = len(results)
        acc, lower, upper = compute_wilson_score_interval(num_correct, num_total)
        accuracy_data[val] = (num_correct, num_total)

        sweep_results.append({
            "control_value": val,
            "num_correct": num_correct,
            "num_total": num_total,
            "accuracy": round(acc, 4),
            "lower_95_ci": round(lower, 4),
            "upper_95_ci": round(upper, 4)
        })

    boundary = estimate_operational_boundary(list(accuracy_data.keys()), accuracy_data, tau=tau, harder_when=harder_when)

    return {
        "task_name": task_name,
        "tau": tau,
        "harder_when": harder_when,
        "estimated_operational_boundary": boundary,
        "sweep_results": sweep_results
    }
