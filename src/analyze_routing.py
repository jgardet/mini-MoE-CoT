"""
analyze_routing.py — Visualize MoE expert routing behavior.

One of the most pedagogically interesting aspects of MoE is watching
experts specialize over training. This script:

  1. Runs a set of probe prompts through the trained model
  2. Records which expert(s) activate for each token
  3. Prints a heatmap of expert utilization per prompt category
  4. Shows per-expert activation patterns

Run after training:
  python analyze_routing.py --checkpoint checkpoints/checkpoint-300

Expected behavior after sufficient training:
  - Expert 0 might specialize on math/numbers
  - Expert 1 on planning/tool-use tokens
  - Expert 2 on natural language / synthesis
  - Expert 3 on structured output (<answer>, <think> tags)

  This specialization is emergent — it's not explicitly supervised.
  The router learns it purely from the training signal.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent))

console = Console()

# Probe prompts designed to test different capabilities
PROBE_PROMPTS = {
    "math": [
        "Calculate 15% of 847.",
        "What is the square root of 144 plus 3 cubed?",
        "If 40% of 200 students passed, how many failed?",
    ],
    "tool_planning": [
        "<tool>calc(3 * 4 + 2)</tool>",
        "<tool>search(population of Switzerland)</tool>",
        "<think>I need to use the calculator to find this.</think>",
    ],
    "reasoning": [
        "<think>Let me break this into steps. First, I'll identify the key variables.",
        "All roses are flowers. Some flowers fade. Therefore...",
        "Step 1: Define the problem. Step 2: Identify constraints.",
    ],
    "structured_output": [
        "<answer>The result is 42.</answer>",
        "<think>Computing now...</think><answer>Done.</answer>",
        "The final answer to the question is:",
    ],
}


def get_expert_activations(model, tokenizer, text: str, device: torch.device) -> dict:
    """Run a forward pass and capture which experts activate for each token.

    We hook into the MoE layer's router to intercept the routing decisions
    without modifying the forward pass behavior.

    Args:
        model: MiniMoEStudent with trained MoE layer.
        tokenizer: Tokenizer.
        text: Input text to analyze.
        device: CUDA/CPU device.

    Returns:
        dict with:
          "tokens": list of decoded token strings
          "expert_indices": (seq_len, top_k) expert assignments per token
          "routing_weights": (seq_len, top_k) softmax weights per token
          "expert_counts": (num_experts,) how many tokens each expert handled
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Storage for routing info (captured via hook)
    routing_info = {}

    def routing_hook(module, input, output):
        """Hook that captures router output during forward pass."""
        # output from Router.forward = (weights, indices, logits)
        weights, indices, logits = output
        routing_info["weights"] = weights.detach().cpu()
        routing_info["indices"] = indices.detach().cpu()

    # Register hook on the router
    hook = model.moe_layer.router.register_forward_hook(routing_hook)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    hook.remove()

    # Decode tokens for display
    tokens = [
        tokenizer.decode([tid]) for tid in input_ids[0].tolist()
    ]

    # Expert counts: how many tokens each expert handled (at any top-k position)
    num_experts = model.moe_layer.num_experts
    expert_counts = torch.zeros(num_experts)
    indices = routing_info["indices"][0]  # (seq_len, top_k)
    for k_idx in range(indices.shape[1]):
        for expert_id in range(num_experts):
            expert_counts[expert_id] += (indices[:, k_idx] == expert_id).sum()

    return {
        "tokens": tokens,
        "expert_indices": indices,
        "routing_weights": routing_info["weights"][0],
        "expert_counts": expert_counts,
        "text": text,
    }


def analyze_category(
    model,
    tokenizer,
    category: str,
    prompts: list[str],
    device: torch.device,
) -> dict:
    """Analyze expert routing for a category of prompts.

    Args:
        model: Trained MiniMoEStudent.
        tokenizer: Tokenizer.
        category: Category name (for display).
        prompts: List of probe prompts.
        device: Device.

    Returns:
        Aggregated expert utilization stats.
    """
    num_experts = model.moe_layer.num_experts
    total_counts = torch.zeros(num_experts)

    for prompt in prompts:
        result = get_expert_activations(model, tokenizer, prompt, device)
        total_counts += result["expert_counts"]

    # Normalize to percentages
    total = total_counts.sum()
    percentages = (total_counts / total * 100) if total > 0 else total_counts

    return {
        "category": category,
        "expert_percentages": percentages,
        "total_tokens": int(total.item()),
    }


def print_routing_heatmap(results: list[dict], num_experts: int):
    """Print a rich table showing expert utilization per category.

    Args:
        results: List of category analysis results.
        num_experts: Number of experts.
    """
    table = Table(title="Expert Routing Heatmap (% of tokens per expert)")
    table.add_column("Category", style="bold cyan", width=20)
    table.add_column("Tokens", justify="right", style="dim")

    for e in range(num_experts):
        table.add_column(f"Expert {e}", justify="right")

    for result in results:
        percs = result["expert_percentages"]
        # Color the highest expert green
        max_expert = percs.argmax().item()
        row = [result["category"], str(result["total_tokens"])]
        for e in range(num_experts):
            pct = percs[e].item()
            bar = "█" * int(pct / 5)  # 1 block per 5%
            cell = f"{bar} {pct:.1f}%"
            if e == max_expert:
                cell = f"[green]{cell}[/]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


def print_token_level_routing(result: dict, max_tokens: int = 20):
    """Print which expert each token was routed to (first N tokens).

    Args:
        result: Output from get_expert_activations.
        max_tokens: Maximum tokens to display.
    """
    table = Table(title=f"Token-level routing: '{result['text'][:40]}...'")
    table.add_column("Token", style="cyan", width=15)
    table.add_column("Expert 1 (w)", justify="center")
    table.add_column("Expert 2 (w)", justify="center")

    tokens = result["tokens"]
    indices = result["expert_indices"]
    weights = result["routing_weights"]

    for i in range(min(len(tokens), max_tokens)):
        tok = repr(tokens[i])
        e1 = indices[i, 0].item()
        e2 = indices[i, 1].item()
        w1 = weights[i, 0].item()
        w2 = weights[i, 1].item()
        table.add_row(
            tok,
            f"E{e1} ({w1:.2f})",
            f"E{e2} ({w2:.2f})",
        )

    console.print(table)


def main():
    import typer

    app = typer.Typer()

    @app.command()
    def run(
        checkpoint: str = typer.Option(
            None, "--checkpoint", "-c",
            help="Path to checkpoint directory"
        ),
        token_level: bool = typer.Option(
            False, "--token-level",
            help="Show token-level routing for first probe"
        ),
    ):
        """Analyze MoE expert routing behavior on probe prompts."""
        from src.infer import load_trained_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_trained_model(checkpoint)
        model.eval()

        console.print(Panel(
            "[bold]Analyzing expert routing behavior[/]\n"
            f"Model: {type(model).__name__}\n"
            f"Experts: {model.moe_layer.num_experts} total, "
            f"top-{model.moe_layer.top_k} active",
            title="🔍 Routing Analysis",
            border_style="cyan",
        ))

        # Analyze each category
        results = []
        for category, prompts in PROBE_PROMPTS.items():
            result = analyze_category(model, tokenizer, category, prompts, device)
            results.append(result)
            console.print(f"  [dim]Analyzed category: {category}[/]")

        # Print heatmap
        console.print()
        print_routing_heatmap(results, model.moe_layer.num_experts)

        # Token-level detail for first prompt in each category
        if token_level:
            console.print("\n[bold]Token-level routing detail:[/]")
            for category, prompts in list(PROBE_PROMPTS.items())[:2]:
                detail = get_expert_activations(
                    model, tokenizer, prompts[0], device
                )
                print_token_level_routing(detail)

        console.print(
            "\n[dim]Tip: Run before and after training to see expert specialization emerge.[/]"
        )

    app()


if __name__ == "__main__":
    main()
