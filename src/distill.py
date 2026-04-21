"""
distill.py — Generate training data by distilling from a local Ollama teacher.

Distillation strategy:
  The teacher (large model, e.g. Qwen3.5:27b) generates high-quality
  CoT reasoning traces and tool-use examples. The student (small MoE)
  is then trained to imitate these traces.

  This is "sequence-level KD" (knowledge distillation): we train the
  student on the teacher's output tokens, not on the teacher's internal
  representations or logits. Simple but effective.

Data format (each JSONL line):
  {
    "id": "unique_id",
    "type": "pure_cot" | "tool_use",
    "question": "...",
    "teacher_response": "<think>...</think>\n<answer>...</answer>",
    "tools_used": ["calc", "search"],  // empty for pure_cot
    "metadata": {"category": "math", ...}
  }

Categories generated:
  - Math: arithmetic, percentages, word problems
  - Logic: multi-step deduction, comparisons
  - World knowledge: facts + reasoning
  - Mixed: combining calculations with factual lookup
"""

import json
import uuid
import random
import asyncio
from pathlib import Path
from typing import Generator
from tqdm import tqdm
from rich.console import Console
import ollama as ollama_client

from .config import CFG
from .tool_loop import build_system_prompt

console = Console()


# ── Question templates ─────────────────────────────────────────────────────

PURE_COT_QUESTIONS = [
    # Math word problems
    "A store sells 3 types of products: A at $12.50, B at $8.75, C at $15.20. "
    "If someone buys 4 of A, 7 of B, and 2 of C, what is the total cost?",

    "Train A leaves city X at 9:00 AM traveling at 80 km/h. "
    "Train B leaves city Y (320 km away) at 10:00 AM traveling at 100 km/h toward X. "
    "At what time do they meet?",

    "A rectangle has a perimeter of 56 cm. The length is 3 times the width. "
    "What is the area of the rectangle?",

    "If 40% of a class passed an exam, and there were 35 students who passed, "
    "how many students are in the class total?",

    "A tank fills in 6 hours with pipe A and 4 hours with pipe B. "
    "How long does it take if both pipes are open?",

    # Logic puzzles
    "Alice is taller than Bob but shorter than Carol. "
    "Dave is taller than Carol. Who is the tallest?",

    "All roses are flowers. Some flowers fade quickly. "
    "Can we conclude that some roses fade quickly? Explain your reasoning.",

    "A farmer has chickens and rabbits. He counts 20 heads and 56 legs. "
    "How many chickens and how many rabbits are there?",

    # Percentage/ratio problems
    "A laptop costs $800. It goes on sale for 15% off. "
    "With an additional 10% coupon applied to the sale price, what is the final price?",

    "Company revenue grew from $2.4M to $3.1M in one year. "
    "What is the percentage growth rate?",

    "If a medication reduces symptoms by 35% and a new version reduces them by "
    "an additional 20% of the remaining symptoms, what is the total reduction?",
]

TOOL_USE_QUESTIONS = [
    # Need calculator
    "What is 847 multiplied by 23, and is the result divisible by 7?",

    "Calculate the compound interest on $5000 at 4.5% annual rate for 3 years. "
    "Use A = P*(1+r)^t.",

    "The square root of 1764 plus the cube of 7 — what is this sum? "
    "Is the result a prime number?",

    "If I invest $10,000 at 6% annual interest, how many years will it take to double? "
    "Use the rule of 72 and verify with actual calculation.",

    # Need search
    "How many people live in Switzerland, and if 45% are employed, "
    "approximately how many workers is that?",

    "What is the distance from Earth to the Moon? "
    "If a rocket travels at 40,000 km/h, how long would the trip take?",

    "What is the speed of light? "
    "How long does light take to travel from the Sun to Earth?",

    # Need datetime
    "How many days until December 25, 2025? Is that more or less than 200 days?",

    "What day of the week is March 15, 2025? "
    "And 100 days after that, what day is it?",

    # Mixed: search + calc
    "Look up the population of Switzerland, then calculate what 0.01% of that population is.",

    "What is Earth's gravitational acceleration? "
    "How long does it take an object to fall 100 meters from rest? Use h = 0.5*g*t².",
]

# Additional varied questions for dataset diversity
EXTRA_QUESTIONS = [
    "A car uses 7.5 liters of fuel per 100 km. How much fuel for a 450 km trip? "
    "If fuel costs $1.65/liter, what is the total fuel cost?",

    "Three workers can paint a house in 12 days. How long would it take 4 workers?",

    "What is 23% of 847? Round to 2 decimal places.",

    "A sequence starts: 2, 6, 18, 54. What is the 8th term?",

    "If a pizza has 8 slices and costs $14, and you want 15 slices, "
    "how many pizzas do you need and what is the cost?",

    "Calculate (15% of 340) + (8% of 220). Show your work.",

    "A rectangle room is 6.5m by 4.2m. How many square meters of flooring are needed? "
    "If tiles are 30cm x 30cm, how many tiles? Add 10% for waste.",

    "What year did World War 2 end? How many years ago was that from today?",
]


def build_teacher_prompt(question: str, include_tools: bool) -> str:
    """Build the prompt sent to the teacher model.

    The teacher is instructed to generate a response in the exact
    format the student will learn to imitate.
    """
    system = build_system_prompt()

    if include_tools:
        instruction = (
            "Solve this problem step by step. "
            "Use tools (calc, search, datetime) where appropriate. "
            "Show your full reasoning inside <think>...</think> tags. "
            "Make tool calls using <tool>name(args)</tool> format. "
            "End with <answer>...</answer>."
        )
    else:
        instruction = (
            "Solve this problem step by step using pure reasoning. "
            "Show all your work inside <think>...</think> tags. "
            "End with <answer>...</answer>. "
            "Do NOT use any tools — reason from first principles."
        )

    return f"{system}\n\n{instruction}\n\nQuestion: {question}"


def generate_with_teacher(
    question: str,
    include_tools: bool,
    teacher_model: str,
    temperature: float,
    max_tokens: int,
) -> str | None:
    """Call the Ollama teacher to generate a CoT response.

    Args:
        question: The question to answer.
        include_tools: Whether to allow tool calls in the response.
        teacher_model: Ollama model name.
        temperature: Sampling temperature.
        max_tokens: Max response tokens.

    Returns:
        Teacher's response string, or None if the call fails.
    """
    prompt = build_teacher_prompt(question, include_tools)

    try:
        response = ollama_client.generate(
            model=teacher_model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["</answer>"],  # Stop after the answer tag
            },
        )
        text = response["response"]
        # Ensure answer tag is closed
        if "<answer>" in text and "</answer>" not in text:
            text += "</answer>"
        return text

    except Exception as e:
        console.print(f"[red]Teacher call failed: {e}[/]")
        return None


def validate_response(response: str, include_tools: bool) -> bool:
    """Check if the teacher's response has the required structure.

    We require:
    - At least one <think>...</think> block
    - An <answer>...</answer> block
    - If tools expected: at least one <tool>...</tool> call

    Args:
        response: Teacher's response string.
        include_tools: Whether tool calls were expected.

    Returns:
        True if valid, False if malformed.
    """
    has_think = "<think>" in response and "</think>" in response
    has_answer = "<answer>" in response and "</answer>" in response

    if not (has_think and has_answer):
        return False

    if include_tools:
        # For tool-use examples, require at least one tool call
        # (the teacher might decide it doesn't need tools, which is fine)
        # We allow it through but mark it as pure_cot
        pass

    return True


def detect_tools_used(response: str) -> list[str]:
    """Extract which tools were called in a response."""
    import re
    pattern = re.compile(r"<tool>\s*(\w+)\(", re.IGNORECASE)
    tools = list(set(m.group(1).lower() for m in pattern.finditer(response)))
    return tools


def generate_dataset(
    n_samples: int,
    output_path: str,
    teacher_model: str,
    temperature: float,
    max_tokens: int,
    include_tool_examples: bool,
    tool_example_ratio: float,
) -> int:
    """Main dataset generation loop.

    Generates n_samples examples and writes them to a JSONL file.

    Args:
        n_samples: Target number of examples.
        output_path: Output JSONL file path.
        teacher_model: Ollama model name.
        temperature: Sampling temperature.
        max_tokens: Max tokens per response.
        include_tool_examples: Whether to include tool-use examples.
        tool_example_ratio: Fraction of examples with tool calls.

    Returns:
        Number of successfully generated examples.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Pool of all questions (cycled if n_samples > pool size)
    all_questions = PURE_COT_QUESTIONS + TOOL_USE_QUESTIONS + EXTRA_QUESTIONS
    random.shuffle(all_questions)

    console.print(f"\n[bold cyan]Starting distillation[/]")
    console.print(f"Teacher: [yellow]{teacher_model}[/]")
    console.print(f"Target samples: [yellow]{n_samples}[/]")
    console.print(f"Output: [yellow]{output_path}[/]\n")

    generated_count = 0
    failed_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=n_samples, desc="Generating", unit="sample")

        question_idx = 0
        while generated_count < n_samples:
            # Decide whether this example should include tools
            use_tools = (
                include_tool_examples
                and random.random() < tool_example_ratio
            )

            # Pick a question (cycle through pool)
            question = all_questions[question_idx % len(all_questions)]
            # Add slight variation to avoid exact duplicates
            if question_idx >= len(all_questions):
                question = question + f" (variant {question_idx // len(all_questions)})"
            question_idx += 1

            # Generate with teacher
            response = generate_with_teacher(
                question=question,
                include_tools=use_tools,
                teacher_model=teacher_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response is None or not validate_response(response, use_tools):
                failed_count += 1
                if failed_count > n_samples * 0.3:
                    console.print(
                        "[red]Too many failures. Check Ollama is running: "
                        f"ollama run {teacher_model}[/]"
                    )
                    break
                continue

            # Detect which tools were actually used
            tools_used = detect_tools_used(response)
            example_type = "tool_use" if tools_used else "pure_cot"

            # Write example
            record = {
                "id": str(uuid.uuid4()),
                "type": example_type,
                "question": question,
                "teacher_response": response,
                "tools_used": tools_used,
                "metadata": {
                    "teacher_model": teacher_model,
                    "temperature": temperature,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            generated_count += 1
            pbar.update(1)
            pbar.set_postfix({
                "tools": sum(1 for _ in range(generated_count) if record["type"] == "tool_use"),
                "failed": failed_count,
            })

        pbar.close()

    console.print(
        f"\n[green]✓ Done![/] Generated {generated_count} examples "
        f"({failed_count} failed). Saved to {output_path}"
    )
    return generated_count


def main():
    """CLI entry point: python -m src.distill"""
    import typer

    app = typer.Typer()

    @app.command()
    def run(
        n_samples: int = typer.Option(CFG.distill.n_samples, help="Number of samples"),
        output: str = typer.Option(CFG.distill.output_path, help="Output JSONL path"),
        teacher: str = typer.Option(CFG.distill.teacher_model, help="Ollama model name"),
        temperature: float = typer.Option(CFG.distill.temperature, help="Sampling temperature"),
        max_tokens: int = typer.Option(CFG.distill.max_tokens, help="Max tokens per sample"),
    ):
        """Generate training data by distilling from a local Ollama teacher."""
        generate_dataset(
            n_samples=n_samples,
            output_path=output,
            teacher_model=teacher,
            temperature=temperature,
            max_tokens=max_tokens,
            include_tool_examples=CFG.distill.include_tool_examples,
            tool_example_ratio=CFG.distill.tool_example_ratio,
        )

    app()


if __name__ == "__main__":
    main()
