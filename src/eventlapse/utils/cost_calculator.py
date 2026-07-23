from typing import Dict, Any, Tuple

# Price dictionary per 1,000,000 tokens: (input_price_per_1M, output_price_per_1M)
MODEL_PRICING_PER_1M: Dict[str, Tuple[float, float]] = {
    # Gemini
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-thinking": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.5-flash": (0.10, 0.40),
    "gemini-3.5-flash": (0.10, 0.40),

    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),

    # Anthropic
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku": (1.00, 5.00),
    "claude-3-5-haiku-20241022": (1.00, 5.00),

    # Open-Source / vLLM (Local / Self-hosted)
    "vllm": (0.00, 0.00),
}

def calculate_api_cost(
    model_name: str,
    prompt_tokens: int,
    candidate_tokens: int,
    provider: str = ""
) -> float:
    """
    Calculates estimated API cost in USD based on input (prompt) and output (candidate) token counts.
    """
    if provider.lower() == "vllm":
        return 0.0

    model_key = model_name.lower().strip()

    # Find matching key
    matched_pricing = None
    for k, price_tuple in MODEL_PRICING_PER_1M.items():
        if k in model_key:
            matched_pricing = price_tuple
            break

    if matched_pricing is None:
        # Default estimated fallback pricing: $1.00/1M prompt, $4.00/1M completion
        matched_pricing = (1.00, 4.00)

    input_price, output_price = matched_pricing
    cost = (prompt_tokens / 1_000_000.0) * input_price + (candidate_tokens / 1_000_000.0) * output_price
    return round(cost, 6)
