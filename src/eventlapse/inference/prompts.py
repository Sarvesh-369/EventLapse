from typing import Dict, Any

PROMPT_TEMPLATES = {
    "direct": """You are an expert video reasoning system.
Question: {question}

Answer the question directly and accurately. Return ONLY your final answer formatted cleanly.""",

    "structured_trace": """You are an expert video reasoning system.
Question: {question}

Provide a detailed structured evidence ledger tracking all events, timestamps, object identities, and state changes step-by-step. Then conclude with your final answer.
Format your output as JSON matching the requested schema.""",

    "multi_turn_verification": """Verification Step:
Review the following step-by-step event trace:
{trace_text}

Verify if all visual events, object movements, and timestamps are accurate. Correct any discrepancies and output the verified final answer."""
}

def get_prompt_for_condition(condition: str, question: str, extra_ctx: str = "") -> str:
    template = PROMPT_TEMPLATES.get(condition, PROMPT_TEMPLATES["direct"])
    return template.format(question=question, trace_text=extra_ctx)
