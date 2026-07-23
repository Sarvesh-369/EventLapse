from typing import Dict, Any, List

def run_prompting_intervention_experiment(condition_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary = {}
    for cond, results in condition_results.items():
        correct = sum(1 for r in results if r.get("exact_match_result", False))
        total = len(results)
        summary[cond] = {
            "accuracy": round(correct / max(1, total), 4),
            "num_total": total
        }
    return summary
