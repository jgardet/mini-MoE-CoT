"""
infer.py — CLI entry point for running inference on the trained model.

Usage examples:
  # Basic question
  python -m src.infer --prompt "What is 15% of 847?"

  # With tool use visible
  python -m src.infer --prompt "What is the population of Switzerland times 0.01%?" --show-tools

  # Quiet mode (just the answer)
  python -m src.infer --prompt "2 + 2?" --quiet

  # Load specific checkpoint
  python -m src.infer --checkpoint checkpoints/checkpoint-300 --prompt "..."

  # Interactive REPL mode
  python -m src.infer --interactive
"""

import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from .config import CFG
from .model import build_model, MiniMoEStudent
from .tool_loop import ToolDispatcher, ToolUseInferenceLoop

console = Console()


def load_trained_model(checkpoint_path: str | None = None):
    """Load the model, optionally from a checkpoint.

    If checkpoint_path is provided and exists, loads saved MoE + LoRA weights.
    Otherwise, loads the base model with randomly initialized MoE (for testing).

    Args:
        checkpoint_path: Path to checkpoint directory, or None.

    Returns:
        (model, tokenizer) tuple.
    """
    model, tokenizer = build_model()

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt_dir = Path(checkpoint_path)

        # Load MoE layer weights
        moe_weights = ckpt_dir / "moe_layer.pt"
        if moe_weights.exists():
            state_dict = torch.load(moe_weights, map_location="cpu")
            model.moe_layer.load_state_dict(state_dict)
            console.print(f"[green]✓ MoE weights loaded from {moe_weights}[/]")

        # Load LoRA adapters
        lora_dir = ckpt_dir / "lora_adapters"
        if lora_dir.exists():
            from peft import PeftModel
            model.base_model = PeftModel.from_pretrained(
                model.base_model.base_model,
                str(lora_dir),
            )
            console.print(f"[green]✓ LoRA adapters loaded from {lora_dir}[/]")
    else:
        console.print(
            "[yellow]⚠ No checkpoint found. Using base model + random MoE.[/]\n"
            "[dim]Run `python -m src.train` first for best results.[/]"
        )

    model.eval()
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    show_thinking: bool = True,
) -> dict:
    """Run a single inference with the tool loop.

    Args:
        model: Loaded MiniMoEStudent.
        tokenizer: Tokenizer.
        prompt: User question.
        show_thinking: Whether to show <think> blocks.

    Returns:
        Result dict from ToolUseInferenceLoop.
    """
    dispatcher = ToolDispatcher()
    loop = ToolUseInferenceLoop(model, tokenizer, dispatcher)
    return loop.run(prompt, show_thinking=show_thinking)


def interactive_repl(model, tokenizer):
    """Run an interactive REPL (Read-Eval-Print Loop).

    The user types questions one at a time and sees the model's
    reasoning process in real time.
    """
    console.print(Panel(
        "[bold]Mini-MoE-CoT Interactive Mode[/]\n"
        "Type your question and press Enter.\n"
        "Commands: [cyan]:quit[/] to exit, [cyan]:notthink[/] to hide reasoning.",
        title="🤖 Interactive Mode",
        border_style="blue",
    ))

    show_thinking = True

    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/]")
            break

        if not user_input:
            continue

        if user_input == ":quit":
            console.print("[yellow]Goodbye![/]")
            break

        if user_input == ":notthink":
            show_thinking = not show_thinking
            console.print(
                f"[cyan]Thinking display: {'ON' if show_thinking else 'OFF'}[/]"
            )
            continue

        result = run_inference(model, tokenizer, user_input, show_thinking)

        if not show_thinking:
            # Just print the final answer cleanly
            console.print(f"\n[bold green]Answer:[/] {result['answer']}")

        # Show tool call summary
        if result["tool_calls"]:
            console.print(
                f"[dim]Used {len(result['tool_calls'])} tool call(s) "
                f"over {result['rounds']} round(s)[/]"
            )


def main():
    """CLI entry point: python -m src.infer"""
    import typer

    app = typer.Typer(help="Run inference on the Mini-MoE-CoT student model.")

    @app.command()
    def run(
        prompt: str = typer.Option(None, "--prompt", "-p", help="Question to answer"),
        checkpoint: str = typer.Option(None, "--checkpoint", "-c", help="Checkpoint directory"),
        show_thinking: bool = typer.Option(True, "--show-thinking/--quiet", help="Show CoT traces"),
        interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive REPL mode"),
    ):
        """Run inference with the trained Mini-MoE-CoT model."""

        model, tokenizer = load_trained_model(checkpoint)

        if interactive:
            interactive_repl(model, tokenizer)
        elif prompt:
            result = run_inference(model, tokenizer, prompt, show_thinking)
            if not show_thinking:
                console.print(f"\n[bold]Answer:[/] {result['answer']}")
        else:
            console.print(
                "[red]Please provide --prompt or --interactive flag.[/]\n"
                "Example: python -m src.infer --prompt 'What is 15% of 847?'"
            )

    app()


if __name__ == "__main__":
    main()
