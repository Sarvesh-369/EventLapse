from typing import List, Dict, Any, Optional, Tuple
from eventlapse.evaluation.exact_match import compute_wilson_score_interval

def estimate_operational_boundary(
    param_values: List[float],
    accuracy_data: Dict[float, Tuple[int, int]], # param_val -> (num_correct, num_total)
    tau: float = 0.80,
    harder_when: str = "increasing"
) -> Optional[float]:
    """
    Operational capability boundary: maximum difficulty for which the lower 95% confidence bound is >= tau.
    """
    sorted_params = sorted(param_values)
    if harder_when == "decreasing":
        # Harder when parameter value is decreasing (e.g. duration ratio r)
        sorted_params = sorted(param_values, reverse=True)

    last_reliable_param = None

    for val in sorted_params:
        correct, total = accuracy_data.get(val, (0, 0))
        acc, lower_ci, upper_ci = compute_wilson_score_interval(correct, total)
        if lower_ci >= tau:
            last_reliable_param = val
        else:
            break

    return last_reliable_param
