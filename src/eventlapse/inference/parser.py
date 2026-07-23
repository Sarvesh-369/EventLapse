import json
import re
from typing import Dict, Any, Optional, Tuple

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # Try direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Search for markdown codeblock ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Search for outermost JSON object { ... }
    match_raw = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_raw:
        try:
            return json.loads(match_raw.group(1))
        except Exception:
            pass

    return None

def normalize_final_answer(raw_str: str) -> str:
    cleaned = str(raw_str).strip().lower()
    cleaned = re.sub(r"[^\w\s\.-]", "", cleaned)
    # Remove leading labels like "answer: "
    cleaned = re.sub(r"^(the\s+)?answer\s+(is\s+)?", "", cleaned)
    return cleaned.strip()

def parse_model_response(raw_text: str, parsed_json: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[Dict[str, Any]], bool]:
    valid = False
    predicted_answer = ""
    evidence = parsed_json

    if not evidence:
        evidence = extract_json_from_text(raw_text)

    if evidence and isinstance(evidence, dict) and "final_answer" in evidence:
        predicted_answer = normalize_final_answer(str(evidence["final_answer"]))
        valid = True
    elif raw_text:
        # Fallback regex extraction for direct responses
        predicted_answer = normalize_final_answer(raw_text)
        valid = len(predicted_answer) > 0

    return predicted_answer, evidence, valid
