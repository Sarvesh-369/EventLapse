from typing import Dict, Any, List, Tuple
from eventlapse.evaluation.exact_match import compute_exact_match
from eventlapse.evaluation.error_taxonomy import classify_trace_error

class MorseTraceEvaluator:
    """
    MORSE-style step-by-step trace & error taxonomy evaluator.
    Evaluates:
    1. Step-level alignment: state transitions s_i, events e_i, operations o_i.
    2. Final answer exact match.
    3. Categorized error taxonomy breakdown (Hallucination, Data Omission, Temporal, Logic, Visual, Math).
    """

    def evaluate_sample_trace(
        self,
        predicted_answer: str,
        predicted_trace: Dict[str, Any],
        gt_answer: str,
        gt_trace: Dict[str, Any]
    ) -> Dict[str, Any]:
        is_exact = compute_exact_match(predicted_answer, gt_answer)

        errors = classify_trace_error(predicted_trace, gt_trace, is_exact)

        gt_steps = gt_trace.get("steps", [])
        pred_steps = predicted_trace.get("steps", []) if isinstance(predicted_trace, dict) else []

        step_alignment_score = 0.0
        if len(gt_steps) > 0:
            match_count = min(len(gt_steps), len(pred_steps))
            step_alignment_score = round(match_count / len(gt_steps), 3)

        return {
            "exact_match": is_exact,
            "step_alignment_score": step_alignment_score,
            "detected_error_categories": errors,
            "has_error": not is_exact
        }
