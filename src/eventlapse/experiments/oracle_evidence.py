from typing import Dict, Any, List

def run_oracle_evidence_experiment(oracle_results: List[Dict[str, Any]], baseline_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc_oracle = sum(1 for r in oracle_results if r.get("exact_match_result", False)) / max(1, len(oracle_results))
    acc_base = sum(1 for r in baseline_results if r.get("exact_match_result", False)) / max(1, len(baseline_results))

    return {
        "baseline_accuracy": round(acc_base, 4),
        "oracle_accuracy": round(acc_oracle, 4),
        "accuracy_gain": round(acc_oracle - acc_base, 4)
    }
