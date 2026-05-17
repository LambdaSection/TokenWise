"""Tests for pricing database."""

import pytest

from tokenwise.pricing.models import get_pricing, list_models, update_pricing, ModelPricing


class TestPricing:
    def test_get_pricing_known(self):
        p = get_pricing("gpt-4o")
        assert p.input_per_m == 2.5
        assert p.output_per_m == 10.0

    def test_get_pricing_unknown(self):
        with pytest.raises(ValueError):
            get_pricing("nonexistent-model-xyz")

    def test_provider_detection(self):
        assert get_pricing("gpt-4o").provider == "OpenAI"
        assert get_pricing("claude-sonnet-4-20250514").provider == "Anthropic"
        assert get_pricing("gemini-2.5-pro").provider == "Google"
        assert get_pricing("mistral-large-2").provider == "Mistral"
        assert get_pricing("llama-3.3-70b").provider == "Groq"
        assert get_pricing("deepseek-chat").provider == "DeepSeek"

    def test_list_models(self):
        models = list_models()
        assert len(models) > 10
        assert "gpt-4o" in models

    def test_list_models_by_provider(self):
        openai_models = list_models(provider="OpenAI")
        assert len(openai_models) > 0
        assert all(m.startswith(("gpt-", "o3", "o4")) for m in openai_models)

    def test_update_pricing(self):
        update_pricing("test-model", {"input": 1.0, "output": 2.0})
        p = get_pricing("test-model")
        assert p.input_per_m == 1.0
        assert p.output_per_m == 2.0

    def test_cost_calculation(self):
        p = ModelPricing(model="test", input_per_m=2.0, output_per_m=8.0)
        cost = p.cost(input_tokens=1_000_000, output_tokens=500_000)
        assert cost == 6.0
