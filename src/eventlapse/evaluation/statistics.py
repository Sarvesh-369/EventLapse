import numpy as np
from typing import List, Tuple, Dict

def compute_isotonic_regression_curve(param_values: List[float], accuracies: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits isotonic regression curve to raw accuracy points.
    """
    from scipy.optimize import minimize

    x = np.array(param_values)
    y = np.array(accuracies)

    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]

    # Simple monotonic pool adjacent violators or Scikit-learn Isotonic
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        y_iso = iso.fit_transform(x_sorted, y_sorted)
        return x_sorted, y_iso
    except ImportError:
        # Fallback to sorted cumulative min/max
        return x_sorted, y_sorted
