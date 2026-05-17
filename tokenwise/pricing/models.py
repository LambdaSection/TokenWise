"""LLM pricing database.

Prices are per 1M tokens (USD). Sources: OpenAI, Anthropic, Google, Mistral.
Last updated: 2026-05.
"""

from dataclasses import dataclass, field

PRICING_DB: dict[str, dict[str, float]] = {
    # --- OpenAI ---
    "gpt-4.1": {"input": 2.0, "output": 8.0, "cache_read": 0.5, "cache_write": 2.5},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6, "cache_read": 0.1, "cache_write": 0.5},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4, "cache_read": 0.025, "cache_write": 0.1},
    "gpt-4o": {"input": 2.5, "output": 10.0, "cache_read": 1.25, "cache_write": 2.5},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6, "cache_read": 0.075, "cache_write": 0.15},
    "o3": {"input": 10.0, "output": 40.0, "cache_read": 2.5, "cache_write": 10.0},
    "o3-mini": {"input": 1.1, "output": 4.4, "cache_read": 0.55, "cache_write": 1.1},
    "o4-mini": {"input": 1.1, "output": 4.4, "cache_read": 0.55, "cache_write": 1.1},
    # --- Anthropic ---
    "claude-sonnet-4-20250514": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
    },
    "claude-opus-4-20250514": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.5,
        "cache_write": 18.75,
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
    },
    "claude-3-5-haiku-20241022": {
        "input": 0.8,
        "output": 4.0,
        "cache_read": 0.08,
        "cache_write": 1.0,
    },
    # --- Google ---
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0, "cache_read": 0.31, "cache_write": 2.5},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.6, "cache_read": 0.0375, "cache_write": 0.15},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4, "cache_read": 0.025, "cache_write": 0.1},
    # --- Mistral ---
    "mistral-large-2": {"input": 2.0, "output": 6.0, "cache_read": 0.5, "cache_write": 2.0},
    "mistral-small-3": {"input": 0.1, "output": 0.3, "cache_read": 0.025, "cache_write": 0.1},
    "codestral": {"input": 0.3, "output": 0.9, "cache_read": 0.075, "cache_write": 0.3},
    # --- Groq ---
    "llama-3.3-70b": {"input": 0.59, "output": 0.79, "cache_read": 0.0, "cache_write": 0.0},
    "mixtral-8x7b": {"input": 0.24, "output": 0.24, "cache_read": 0.0, "cache_write": 0.0},
    # --- DeepSeek ---
    "deepseek-chat": {"input": 0.27, "output": 1.1, "cache_read": 0.07, "cache_write": 0.27},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19, "cache_read": 0.14, "cache_write": 0.55},
}


@dataclass
class ModelPricing:
    """Pricing info for a single model."""

    model: str
    input_per_m: float
    output_per_m: float
    cache_read_per_m: float = 0.0
    cache_write_per_m: float = 0.0

    @property
    def provider(self) -> str:
        if self.model.startswith(("gpt-", "o3", "o4")):
            return "OpenAI"
        if self.model.startswith("claude"):
            return "Anthropic"
        if self.model.startswith("gemini"):
            return "Google"
        if self.model.startswith(("mistral", "codestral")):
            return "Mistral"
        if self.model.startswith(("llama", "mixtral")):
            return "Groq"
        if self.model.startswith("deepseek"):
            return "DeepSeek"
        return "Unknown"

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts."""
        return (input_tokens / 1_000_000) * self.input_per_m + (
            output_tokens / 1_000_000
        ) * self.output_per_m


def get_pricing(model: str) -> ModelPricing:
    """Get pricing for a model. Falls back to closest match if exact name not found."""
    if model in PRICING_DB:
        p = PRICING_DB[model]
        return ModelPricing(
            model=model,
            input_per_m=p["input"],
            output_per_m=p["output"],
            cache_read_per_m=p.get("cache_read", 0.0),
            cache_write_per_m=p.get("cache_write", 0.0),
        )

    # Fuzzy fallback: try substring match
    for known_model in PRICING_DB:
        if known_model in model or model in known_model:
            p = PRICING_DB[known_model]
            return ModelPricing(
                model=model,
                input_per_m=p["input"],
                output_per_m=p["output"],
                cache_read_per_m=p.get("cache_read", 0.0),
                cache_write_per_m=p.get("cache_write", 0.0),
            )

    raise ValueError(f"Unknown model: {model}. Use list_models() to see available models.")


def list_models(provider: str | None = None) -> list[str]:
    """List available models, optionally filtered by provider."""
    if provider is None:
        return sorted(PRICING_DB.keys())
    return sorted(
        m
        for m in PRICING_DB
        if ModelPricing(
            model=m,
            input_per_m=PRICING_DB[m]["input"],
            output_per_m=PRICING_DB[m]["output"],
        ).provider.lower()
        == provider.lower()
    )


def update_pricing(model: str, pricing: dict[str, float]) -> None:
    """Update or add pricing for a model."""
    PRICING_DB[model] = pricing
