"""Tests for cost estimation engine."""

import pytest

from tokenwise.core.cost import CostEngine
from tokenwise.pricing.models import get_pricing


class TestCostEngine:
    def test_estimate_gpt4o(self):
        engine = CostEngine()
        est = engine.estimate("gpt-4o", input_tokens=1000, output_tokens=500)
        assert est.model == "gpt-4o"
        assert est.provider == "OpenAI"
        assert est.input_tokens == 1000
        assert est.output_tokens == 500
        assert est.total_tokens == 1500
        assert est.total_cost > 0
        assert est.input_cost > 0
        assert est.output_cost > 0

    def test_estimate_claude(self):
        engine = CostEngine()
        est = engine.estimate("claude-sonnet-4-20250514", input_tokens=2000, output_tokens=1000)
        assert est.provider == "Anthropic"
        assert est.total_cost > 0

    def test_estimate_zero_tokens(self):
        engine = CostEngine()
        est = engine.estimate("gpt-4o", input_tokens=0, output_tokens=0)
        assert est.total_cost == 0.0
        assert est.cost_per_1k == 0.0

    def test_compare_sorted(self):
        engine = CostEngine()
        results = engine.compare(
            ["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash"],
            input_tokens=1000,
            output_tokens=500,
        )
        assert len(results) == 3
        assert results[0].total_cost <= results[1].total_cost <= results[2].total_cost

    def test_batch_estimate(self):
        engine = CostEngine()
        calls = [(1000, 500), (2000, 800), (500, 200)]
        est = engine.batch_estimate("gpt-4o", calls)
        assert est.input_tokens == 3500
        assert est.output_tokens == 1500

    def test_fuzzy_model_match(self):
        engine = CostEngine()
        est = engine.estimate("gpt-4o-2024-05-13", input_tokens=100, output_tokens=50)
        assert est.total_cost > 0
