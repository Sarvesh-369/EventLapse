from typing import Dict, Any, List

ERROR_TAXONOMY_CATEGORIES = [
    "missed_event",
    "hallucinated_event",
    "merged_events",
    "duplicated_event",
    "misordered_event",
    "wrong_timestamp",
    "wrong_onset",
    "wrong_offset",
    "wrong_duration_comparison",
    "missing_causal_edge",
    "hallucinated_causal_edge",
    "wrong_root_cause",
    "wrong_rule",
    "correct_rule_wrong_rollout",
    "off_by_one_prediction",
    "lost_object_identity",
    "correct_trace_wrong_final_answer",
    "incorrect_trace_accidental_correct"
]

def classify_trace_error(
    predicted_trace: Dict[str, Any],
    ground_truth_trace: Dict[str, Any],
    exact_match_correct: bool
) -> List[str]:
    detected_errors = []

    if not predicted_trace or not isinstance(predicted_trace, dict):
        detected_errors.append("missed_event")
        return detected_errors

    # Check for trace events versus GT events length
    gt_events = ground_truth_trace.get("events", [])
    pred_events = predicted_trace.get("detected_contacts", []) or predicted_trace.get("detected_events", []) or predicted_trace.get("observed_swaps", [])

    if isinstance(gt_events, list) and isinstance(pred_events, list):
        if len(pred_events) < len(gt_events):
            detected_errors.append("missed_event")
        elif len(pred_events) > len(gt_events):
            detected_errors.append("hallucinated_event")

    # Trace correctness vs Final Answer correctness
    if exact_match_correct and ("missed_event" in detected_errors or "hallucinated_event" in detected_errors):
        detected_errors.append("incorrect_trace_accidental_correct")

    if not exact_match_correct and not detected_errors:
        detected_errors.append("correct_trace_wrong_final_answer")

    return detected_errors
