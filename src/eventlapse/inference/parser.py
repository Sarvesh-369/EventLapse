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

def extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    # Find all \boxed{...} instances
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        # Return the last boxed match (in case prompt template contains an example box like \boxed{4})
        return matches[-1].strip()
    return None

def normalize_final_answer(raw_str: str) -> str:
    if not raw_str:
        return ""
    cleaned = str(raw_str).strip()
    
    # If boxed match
    boxed = extract_boxed_answer(cleaned)
    if boxed is not None:
        cleaned = boxed

    # Remove markdown formatting
    cleaned = re.sub(r"[*_`]", "", cleaned)

    # Search for explicit final answer patterns
    fa_match = re.search(r"(?:final\s+answer|total\s+count|final\tag|count)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if fa_match:
        return fa_match.group(1).strip()

    # Search for standalone integer/number at the end of the text
    num_matches = re.findall(r"(-?\d+(?:\.\d+)?)", cleaned)
    if num_matches:
        return num_matches[-1].strip()

    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^\w\s\.-]", "", cleaned)
    cleaned = re.sub(r"^(the\s+)?answer\s+(is\s+)?", "", cleaned)
    return cleaned.strip()

def parse_model_response(raw_text: str, parsed_json: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[Dict[str, Any]], bool]:
    valid = False
    predicted_answer = ""
    evidence = parsed_json

    if not evidence:
        evidence = extract_json_from_text(raw_text)

    # 1. Try boxed answer extraction first
    boxed = extract_boxed_answer(raw_text)
    if boxed:
        predicted_answer = normalize_final_answer(boxed)
        valid = len(predicted_answer) > 0

    # 2. Try JSON schema evidence
    if not valid and evidence and isinstance(evidence, dict) and "final_answer" in evidence:
        predicted_answer = normalize_final_answer(str(evidence["final_answer"]))
        valid = len(predicted_answer) > 0

    # 3. Fallback regex extraction from raw text
    if not valid and raw_text:
        predicted_answer = normalize_final_answer(raw_text)
        valid = len(predicted_answer) > 0

    # Ensure evidence is never null if we have raw_text
    if not evidence and raw_text:
        evidence = {
            "reasoning_trace": raw_text,
            "boxed_answer": boxed,
            "extracted_answer": predicted_answer
        }

    return predicted_answer, evidence, valid
