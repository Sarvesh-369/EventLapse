import math
from typing import Tuple, List

def compute_exact_match(predicted: str, ground_truth: str) -> bool:
    p = str(predicted).strip().lower()
    g = str(ground_truth).strip().lower()
    return p == g

def compute_wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score binomial confidence interval.
    Returns (accuracy, lower_bound, upper_bound).
    """
    if n == 0:
        return 0.0, 0.0, 0.0

    p_hat = k / n
    z = 1.95996 # for 95% CI

    denominator = 1 + (z ** 2) / n
    centre = p_hat + (z ** 2) / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + (z ** 2) / (4 * n)) / n)

    lower = max(0.0, (centre - spread) / denominator)
    upper = min(1.0, (centre + spread) / denominator)

    return p_hat, lower, upper
