"""TokenWise CLI — LLM Cost Optimizer."""

import argparse
import sys

from rich.console import Console
from rich.table import Table

from tokenwise.core.cost import CostEngine
from tokenwise.core.optimizer import Optimizer
from tokenwise.core.tokenizer import count_tokens
from tokenwise.pricing.models import list_models, get_pricing
from tokenwise.tracker.usage import UsageTracker

console = Console()


def cmd_estimate(args: argparse.Namespace) -> None:
    """Estimate cost for a single call."""
    engine = CostEngine()
    if args.text:
        input_tokens = count_tokens(args.text, model=args.model)
    else:
        input_tokens = args.input_tokens

    est = engine.estimate(args.model, input_tokens, args.output_tokens)
    console.print()
    console.print(f"[bold cyan]{est}[/bold cyan]")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare costs across models."""
    engine = CostEngine()
    models = args.models
    estimates = engine.compare(models, args.input_tokens, args.output_tokens)

    table = Table(title=f"Cost Comparison — {args.input_tokens:,} in / {args.output_tokens:,} out")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Total Cost ($)", justify="right", style="green")
    table.add_column("$ / 1K tokens", justify="right", style="yellow")

    for est in estimates:
        table.add_row(
            est.model,
            est.provider,
            f"${est.total_cost:.4f}",
            f"${est.cost_per_1k:.4f}",
        )

    console.print()
    console.print(table)


def cmd_models(args: argparse.Namespace) -> None:
    """List available models."""
    models = list_models(provider=args.provider)
    table = Table(title=f"Available Models ({len(models)})")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Input $/1M", justify="right", style="green")
    table.add_column("Output $/1M", justify="right", style="green")

    for m in models:
        p = get_pricing(m)
        table.add_row(m, p.provider, f"${p.input_per_m:.2f}", f"${p.output_per_m:.2f}")

    console.print()
    console.print(table)


def cmd_optimize(args: argparse.Namespace) -> None:
    """Get optimization suggestions."""
    optimizer = Optimizer()
    suggestions = optimizer.analyze(args.model, args.input_tokens, args.output_tokens)

    if not suggestions:
        console.print("\n[yellow]No cheaper alternatives found.[/yellow]")
        return

    table = Table(title="Optimization Suggestions")
    table.add_column("#", style="dim")
    table.add_column("Suggestion", style="cyan")
    table.add_column("Details", style="white")
    table.add_column("Savings", justify="right", style="green")

    for i, s in enumerate(suggestions, 1):
        details = s.description
        if s.suggested_cost is not None:
            details += f" (${s.original_cost:.4f} → ${s.suggested_cost:.4f})"
        table.add_row(str(i), s.title, details, f"{s.savings_pct:.1f}%")

    console.print()
    console.print(table)


def cmd_count(args: argparse.Namespace) -> None:
    """Count tokens in text."""
    text = sys.stdin.read() if args.stdin else args.text
    if not text:
        console.print("[red]No text provided. Use --text or pipe input.[/red]")
        return
    tokens = count_tokens(text, model=args.model)
    console.print(f"\n[cyan]{tokens:,}[/cyan] tokens (model: {args.model})")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tokenwise",
        description="TokenWise — LLM Cost Optimizer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # estimate
    p_est = sub.add_parser("estimate", help="Estimate cost for a call")
    p_est.add_argument("--model", "-m", required=True, help="Model name")
    p_est.add_argument("--input-tokens", "-i", type=int, default=0, help="Input token count")
    p_est.add_argument("--output-tokens", "-o", type=int, required=True, help="Output token count")
    p_est.add_argument("--text", "-t", help="Text to count tokens from (instead of --input-tokens)")
    p_est.set_defaults(func=cmd_estimate)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare costs across models")
    p_cmp.add_argument("models", nargs="+", help="Models to compare")
    p_cmp.add_argument("--input-tokens", "-i", type=int, required=True)
    p_cmp.add_argument("--output-tokens", "-o", type=int, required=True)
    p_cmp.set_defaults(func=cmd_compare)

    # models
    p_mod = sub.add_parser("models", help="List available models")
    p_mod.add_argument("--provider", "-p", help="Filter by provider")
    p_mod.set_defaults(func=cmd_models)

    # optimize
    p_opt = sub.add_parser("optimize", help="Get optimization suggestions")
    p_opt.add_argument("--model", "-m", required=True)
    p_opt.add_argument("--input-tokens", "-i", type=int, required=True)
    p_opt.add_argument("--output-tokens", "-o", type=int, required=True)
    p_opt.set_defaults(func=cmd_optimize)

    # count
    p_cnt = sub.add_parser("count", help="Count tokens in text")
    p_cnt.add_argument("--model", "-m", default="gpt-4o")
    p_cnt.add_argument("--text", "-t", help="Text to count")
    p_cnt.add_argument("--stdin", action="store_true", help="Read from stdin")
    p_cnt.set_defaults(func=cmd_count)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
