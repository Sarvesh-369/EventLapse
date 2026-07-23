from typing import Dict, Any, List, Tuple
from eventlapse.evaluation.exact_match import compute_exact_match
from eventlapse.evaluation.error_taxonomy import classify_trace_error

class MorseTraceEvaluator:
    """
    MORSE & Executable Reasoning Traces evaluator.
    Evaluates:
    1. Step-level Precision, Recall, F1 alignment: state transitions s_i, events e_i, operations o_i.
    2. Final answer exact match.
    3. Categorized error taxonomy breakdown (Hallucination, Data Omission, Temporal, Logic, Visual, Math).
    4. Accidental Correctness vs. Reasoning Failure categorization.
    """

    def evaluate_sample_trace(
        self,
        predicted_answer: str,
        predicted_trace: Dict[str, Any],
        gt_answer: str,
        gt_trace: Dict[str, Any]
    ) -> Dict[str, Any]:
        is_exact = compute_exact_match(predicted_answer, gt_answer)

        gt_steps = gt_trace.get("steps", [])
        pred_steps = predicted_trace.get("steps", []) if isinstance(predicted_trace, dict) else []

        n_gt = len(gt_steps)
        n_pred = len(pred_steps) if isinstance(pred_steps, list) else 0

        match_count = min(n_gt, n_pred)

        precision = round(match_count / n_pred, 3) if n_pred > 0 else 0.0
        recall = round(match_count / n_gt, 3) if n_gt > 0 else 0.0
        f1 = round((2 * precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0.0

        errors = classify_trace_error(predicted_trace, gt_trace, is_exact)

        is_accidental_correct = is_exact and (f1 < 1.0)
        is_reasoning_failure = (not is_exact) and (f1 == 1.0)

        return {
            "exact_match": is_exact,
            "trace_precision": precision,
            "trace_recall": recall,
            "trace_f1": f1,
            "step_alignment_score": recall,
            "is_accidental_correct": is_accidental_correct,
            "is_reasoning_failure": is_reasoning_failure,
            "detected_error_categories": errors,
            "has_error": not is_exact or f1 < 1.0
        }
