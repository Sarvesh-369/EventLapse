from typing import Dict, Any, List, Tuple

def validate_trace_structure(trace_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    if "steps" not in trace_data or not isinstance(trace_data["steps"], list):
        errors.append("Trace missing 'steps' list")
        return False, errors

    for idx, step in enumerate(trace_data["steps"]):
        if not isinstance(step, dict):
            errors.append(f"Step {idx} is not a dict")
            continue
        if "state" not in step:
            errors.append(f"Step {idx} missing 'state'")
        if "event" not in step:
            errors.append(f"Step {idx} missing 'event'")
        if "operation" not in step:
            errors.append(f"Step {idx} missing 'operation'")

    return len(errors) == 0, errors
