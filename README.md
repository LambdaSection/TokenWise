# TokenWise

> **LLM Cost Optimizer** — Estimate, track, and optimize your LLM spending across providers.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

## Features

- **Cost Estimation** — Instant pricing for 20+ models across OpenAI, Anthropic, Google, Mistral, Groq, DeepSeek
- **Model Comparison** — Side-by-side cost comparison to find the cheapest option for your workload
- **Optimization Suggestions** — AI-driven recommendations to reduce spending
- **Token Counting** — Accurate token counts via tiktoken
- **Usage Tracking** — Log API calls and monitor spending over time with per-model and per-day breakdowns
- **CLI + Python API** — Use in scripts or from the command line

## Quick Start

```bash
pip install tokenwise
```

### CLI

```bash
# Estimate cost for a call
tokenwise estimate -m gpt-4o -i 2000 -o 500

# Compare models
tokenwise compare gpt-4o claude-sonnet-4-20250514 gemini-2.5-pro -i 2000 -o 500

# Get optimization suggestions
tokenwise optimize -m gpt-4o -i 5000 -o 1000

# List all available models
tokenwise models

# Filter by provider
tokenwise models -p openai

# Count tokens
tokenwise count -t "Hello, world!" -m gpt-4o
```

### Python API

```python
from tokenwise import CostEngine, Optimizer, UsageTracker

# Estimate cost
engine = CostEngine()
est = engine.estimate("gpt-4o", input_tokens=2000, output_tokens=500)
print(est)

# Compare models
cheapest = engine.compare(
    ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro"],
    input_tokens=2000,
    output_tokens=500,
)

# Get optimization suggestions
optimizer = Optimizer()
suggestions = optimizer.analyze("gpt-4o", input_tokens=5000, output_tokens=1000)
for s in suggestions:
    print(f"{s.title}: {s.savings_pct}% savings")

# Track usage
tracker = UsageTracker(storage_path="usage.json")
tracker.log("gpt-4o", input_tokens=2000, output_tokens=500, cost=0.008, tags=["chat", "prod"])
print(f"Total spent: ${tracker.total_cost()}")
print(tracker.cost_by_model())
```

## Supported Providers & Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, o3, o3-mini, o4-mini |
| Anthropic | claude-sonnet-4, claude-opus-4, claude-3-5-sonnet, claude-3-5-haiku |
| Google | gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash |
| Mistral | mistral-large-2, mistral-small-3, codestral |
| Groq | llama-3.3-70b, mixtral-8x7b |
| DeepSeek | deepseek-chat, deepseek-reasoner |

## Development

```bash
git clone https://github.com/Lemniscate-world/TokenWise.git
cd TokenWise
pip install -e ".[dev]"
pre-commit install
```

Run tests:

```bash
pytest
```

## License

MIT
