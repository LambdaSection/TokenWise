"""Cost calculation engine."""

from dataclasses import dataclass

from tokenwise.pricing.models import ModelPricing, get_pricing


@dataclass
class CostEstimate:
    """Result of a cost estimation."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    cost_per_1k: float

    def __str__(self) -> str:
        return (
            f"{self.model} ({self.provider})\n"
            f"  Input:  {self.input_tokens:,} tokens  -> ${self.input_cost:.4f}\n"
            f"  Output: {self.output_tokens:,} tokens  -> ${self.output_cost:.4f}\n"
            f"  Total:  {self.total_tokens:,} tokens  -> ${self.total_cost:.4f}\n"
            f"  Avg:    ${self.cost_per_1k:.4f} / 1K tokens"
        )


class CostEngine:
    """Estimate costs for LLM API calls."""

    def __init__(self, pricing_db: dict[str, dict[str, float]] | None = None):
        self._pricing_db = pricing_db

    def estimate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate:
        """Estimate cost for a single API call."""
        pricing = get_pricing(model)
        input_cost = (input_tokens / 1_000_000) * pricing.input_per_m
        output_cost = (output_tokens / 1_000_000) * pricing.output_per_m
        total_tokens = input_tokens + output_tokens
        total_cost = input_cost + output_cost
        cost_per_1k = (total_cost / total_tokens * 1_000) if total_tokens else 0.0

        return CostEstimate(
            model=model,
            provider=pricing.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total_cost, 6),
            cost_per_1k=round(cost_per_1k, 6),
        )

    def compare(
        self,
        models: list[str],
        input_tokens: int,
        output_tokens: int,
    ) -> list[CostEstimate]:
        """Compare costs across multiple models, sorted cheapest first."""
        estimates = [self.estimate(m, input_tokens, output_tokens) for m in models]
        return sorted(estimates, key=lambda e: e.total_cost)

    def batch_estimate(
        self,
        model: str,
        calls: list[tuple[int, int]],
    ) -> CostEstimate:
        """Estimate cost for a batch of API calls."""
        total_input = sum(inp for inp, _ in calls)
        total_output = sum(out for _, out in calls)
        return self.estimate(model, total_input, total_output)
