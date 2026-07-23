from typing import Dict, Any, List
from eventlapse.evaluation.morse_evaluator import MorseTraceEvaluator

def run_trace_diagnosis_experiment(results_manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluator = MorseTraceEvaluator()
    error_counts: Dict[str, int] = {}
    total_samples = len(results_manifest)

    for item in results_manifest:
        pred_ans = item.get("predicted_answer", "")
        pred_trace = item.get("model_reported_trace", {})
        gt_ans = item.get("ground_truth_answer", "")
        gt_trace = item.get("ground_truth_trace", {})

        res = evaluator.evaluate_sample_trace(pred_ans, pred_trace, gt_ans, gt_trace)

        for err in res["detected_error_categories"]:
            error_counts[err] = error_counts.get(err, 0) + 1

    return {
        "total_samples": total_samples,
        "error_distribution": error_counts,
        "error_percentages": {k: round(v / max(1, total_samples) * 100, 2) for k, v in error_counts.items()}
    }
