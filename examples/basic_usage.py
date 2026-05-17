"""TokenWise basic usage examples."""

from tokenwise import CostEngine, Optimizer, UsageTracker
from tokenwise.core.tokenizer import count_tokens

# --- Cost Estimation ---
print("=" * 60)
print("Cost Estimation")
print("=" * 60)

engine = CostEngine()
est = engine.estimate("gpt-4o", input_tokens=2000, output_tokens=500)
print(est)
print()

# --- Model Comparison ---
print("=" * 60)
print("Model Comparison (2K in / 500 out)")
print("=" * 60)

models = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514", "gemini-2.5-flash", "deepseek-chat"]
results = engine.compare(models, input_tokens=2000, output_tokens=500)
for r in results:
    print(f"  {r.model:35s} ${r.total_cost:.4f}")
print()

# --- Optimization ---
print("=" * 60)
print("Optimization Suggestions (5K in / 1K out on gpt-4o)")
print("=" * 60)

optimizer = Optimizer()
suggestions = optimizer.analyze("gpt-4o", input_tokens=5000, output_tokens=1000)
for i, s in enumerate(suggestions[:5], 1):
    print(f"  {i}. {s.title} ({s.savings_pct}% savings)")
print()

# --- Token Counting ---
print("=" * 60)
print("Token Counting")
print("=" * 60)

text = "The quick brown fox jumps over the lazy dog."
tokens = count_tokens(text, model="gpt-4o")
print(f"  '{text}' = {tokens} tokens")
print()

# --- Usage Tracking ---
print("=" * 60)
print("Usage Tracking")
print("=" * 60)

tracker = UsageTracker()
tracker.log("gpt-4o", input_tokens=2000, output_tokens=500, cost=0.008, tags=["chat"])
tracker.log("gpt-4o-mini", input_tokens=500, output_tokens=200, cost=0.0002, tags=["summarize"])
tracker.log(
    "claude-sonnet-4-20250514",
    input_tokens=3000,
    output_tokens=1500,
    cost=0.0315,
    tags=["analysis"],
)

print(f"  Total cost:    ${tracker.total_cost()}")
print(f"  Total tokens:  {tracker.total_tokens()}")
print(f"  Cost by model: {tracker.cost_by_model()}")
