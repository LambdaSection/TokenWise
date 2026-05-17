"""Cost optimization suggestions."""

from dataclasses import dataclass

from tokenwise.core.cost import CostEngine, CostEstimate
from tokenwise.pricing.models import list_models


@dataclass
class Suggestion:
    """A cost optimization suggestion."""

    title: str
    description: str
    savings_pct: float
    suggested_model: str | None = None
    original_cost: float | None = None
    suggested_cost: float | None = None


class Optimizer:
    """Suggest cost optimizations for LLM usage."""

    def __init__(self) -> None:
        self.engine = CostEngine()

    def suggest_cheaper_model(
        self,
        current_model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> list[Suggestion]:
        """Find cheaper alternatives for the same task."""
        current = self.engine.estimate(current_model, input_tokens, output_tokens)
        all_models = list_models()
        suggestions: list[Suggestion] = []

        for alt in all_models:
            if alt == current_model:
                continue
            alt_est = self.engine.estimate(alt, input_tokens, output_tokens)
            if alt_est.total_cost < current.total_cost:
                savings = (
                    ((current.total_cost - alt_est.total_cost) / current.total_cost * 100)
                    if current.total_cost > 0
                    else 0
                )
                suggestions.append(
                    Suggestion(
                        title=f"Switch to {alt}",
                        description=f"{alt_est.provider} model with similar capabilities",
                        savings_pct=round(savings, 1),
                        suggested_model=alt,
                        original_cost=current.total_cost,
                        suggested_cost=alt_est.total_cost,
                    )
                )

        return sorted(suggestions, key=lambda s: s.savings_pct, reverse=True)[:5]

    def suggest_input_reduction(
        self,
        current_model: str,
        input_tokens: int,
        output_tokens: int,
        reduction_pct: float = 30.0,
    ) -> Suggestion | None:
        """Estimate savings from reducing input tokens (e.g., shorter prompts)."""
        current = self.engine.estimate(current_model, input_tokens, output_tokens)
        reduced_input = int(input_tokens * (1 - reduction_pct / 100))
        reduced = self.engine.estimate(current_model, reduced_input, output_tokens)
        savings = (
            ((current.total_cost - reduced.total_cost) / current.total_cost * 100)
            if current.total_cost > 0
            else 0
        )

        return Suggestion(
            title=f"Reduce input tokens by {reduction_pct:.0f}%",
            description="Trim system prompts, remove redundant context, use concise instructions",
            savings_pct=round(savings, 1),
            original_cost=current.total_cost,
            suggested_cost=reduced.total_cost,
        )

    def analyze(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> list[Suggestion]:
        """Get all optimization suggestions for a usage pattern."""
        suggestions: list[Suggestion] = []
        suggestions.extend(self.suggest_cheaper_model(model, input_tokens, output_tokens))
        input_suggestion = self.suggest_input_reduction(model, input_tokens, output_tokens)
        if input_suggestion:
            suggestions.append(input_suggestion)
        return suggestions
