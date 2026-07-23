from typing import Dict, Any

PROMPTING_CONDITIONS = ["direct", "structured_trace", "multi_turn_verification", "thinking"]

def apply_prompt_intervention(condition: str, question: str) -> Dict[str, Any]:
    if condition not in PROMPTING_CONDITIONS:
        raise ValueError(f"Unknown prompting condition: {condition}")
    return {"prompt_condition": condition, "question": question}
