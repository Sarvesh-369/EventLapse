from typing import Dict, Any

PROMPT_TEMPLATES = {
    "direct": """Question: {question}

Observe the video carefully. Provide your final answer formatted inside \\boxed{{}} (e.g. \\boxed{{4}}).""",

    "structured_trace": """Question: {question}

Observe the video carefully and provide a step-by-step reasoning trace tracking every event, its timestamp (in seconds), and the running count.
Format your response as follows:
1. Step-by-Step Event Ledger (timestamp, event type, running count)
2. Final Answer inside \\boxed{{}} (e.g. \\boxed{{4}})""",

    "multi_turn_verification": """Question: {question}

First, carefully list every detected event with its approximate timestamp and current count.
Second, audit your event list to verify whether any events were missed, merged, or double-counted.
Finally, state your verified final answer inside \\boxed{{}} (e.g. \\boxed{{4}}).""",

    "thinking": """Question: {question}

Use deep step-by-step video analysis. Thoroughly analyze the temporal video sequence frame-by-frame, count all target visual events, and output your final count inside \\boxed{{}} (e.g. \\boxed{{4}}).""",

    "role_prompting": """System: You are an expert video analytics systems auditor specializing in fine-grained temporal event verification.
Question: {question}

Apply rigorous visual event auditing. Log all timestamped visual transitions and state your final answer inside \\boxed{{}} (e.g. \\boxed{{4}})."""
}

def get_prompt_for_condition(condition: str, question: str, extra_ctx: str = "") -> str:
    template = PROMPT_TEMPLATES.get(condition, PROMPT_TEMPLATES["structured_trace"])
    return template.format(question=question, trace_text=extra_ctx)
