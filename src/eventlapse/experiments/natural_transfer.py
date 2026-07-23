from typing import Dict, Any, List

def run_natural_transfer_experiment(repcount_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = sum(1 for r in repcount_results if r.get("exact_match_result", False))
    total = len(repcount_results)
    return {
        "dataset": "repcount_a",
        "num_samples": total,
        "exact_match_accuracy": round(correct / max(1, total), 4)
    }
