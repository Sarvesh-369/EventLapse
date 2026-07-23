from typing import Dict, Any, List

def run_frame_density_experiment(mode_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary = {}
    for mode, results in mode_results.items():
        correct = sum(1 for r in results if r.get("exact_match_result", False))
        total = len(results)
        summary[mode] = {
            "num_correct": correct,
            "num_total": total,
            "accuracy": round(correct / max(1, total), 4)
        }
    return summary
