"""
tool_loop.py — Tool dispatcher and multi-step inference loop.

This module handles the "agentic" part of the pipeline:
  1. The model generates text containing <tool>name(args)</tool> tags
  2. The ToolDispatcher parses and executes the tool call
  3. The result is injected into the context as <tool_result>...</tool_result>
  4. The model continues generating until it produces <answer>...</answer>
     or reaches the max_tool_rounds limit

This implements the "ReAct" pattern (Reasoning + Acting):
  Thought → Action (tool call) → Observation (tool result) → Thought → ...

Special tokens used:
  <think>...</think>     — Internal chain-of-thought reasoning (not shown to user
                           unless show_thinking=True)
  <tool>name(args)</tool> — Tool call request
  <tool_result>...</tool_result> — Tool execution result (injected by dispatcher)
  <answer>...</answer>   — Final answer (stops the loop)

Reference: "ReAct: Synergizing Reasoning and Acting in Language Models"
           (Yao et al., 2022)
"""

import re
import importlib
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from .config import CFG

console = Console()

# Registry of available tools.
# Add new tools by adding a module to tools/ and registering here.
TOOL_REGISTRY = {
    "calc": "tools.calculator",
    "calculator": "tools.calculator",
    "search": "tools.search",
    "datetime": "tools.datetime_tool",
    "date": "tools.datetime_tool",
    "time": "tools.datetime_tool",
}

# Regex patterns for parsing model output
TOOL_CALL_PATTERN = re.compile(
    r"<tool>\s*(\w+)\(([^)]*)\)\s*</tool>",
    re.IGNORECASE | re.DOTALL,
)
ANSWER_PATTERN = re.compile(
    r"<answer>(.*?)</answer>",
    re.IGNORECASE | re.DOTALL,
)
THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.IGNORECASE | re.DOTALL,
)


class ToolDispatcher:
    """Parses and executes tool calls from model output.

    Maintains a call history for debugging and logging.
    """

    def __init__(self):
        self.call_history: list[dict] = []

    def dispatch(self, tool_name: str, args: str) -> str:
        """Execute a named tool with the given arguments.

        Args:
            tool_name: Tool name as it appears in the model's output.
            args: Raw argument string from <tool>name(args)</tool>.

        Returns:
            Tool result as a string.
        """
        module_path = TOOL_REGISTRY.get(tool_name.lower())

        if module_path is None:
            error = f"Unknown tool: '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"
            console.print(f"[red]⚠ {error}[/]")
            return f"<error>{error}</error>"

        try:
            module = importlib.import_module(module_path)
            result = module.run(args.strip())
            self.call_history.append({
                "tool": tool_name,
                "args": args.strip(),
                "result": result,
            })
            return result
        except Exception as e:
            error = f"Tool '{tool_name}' failed: {str(e)}"
            console.print(f"[red]⚠ {error}[/]")
            return f"<error>{error}</error>"

    def parse_tool_calls(self, text: str) -> list[tuple[str, str]]:
        """Extract all tool calls from a text string.

        Args:
            text: Raw model output.

        Returns:
            List of (tool_name, args) tuples.
        """
        return [
            (match.group(1), match.group(2))
            for match in TOOL_CALL_PATTERN.finditer(text)
        ]

    def inject_results(self, text: str) -> str:
        """Replace <tool>calls</tool> with their results in the text.

        Args:
            text: Raw model output containing tool calls.

        Returns:
            Text with tool calls replaced by <tool_result>...</tool_result>.
        """
        def replace_call(match):
            tool_name = match.group(1)
            args = match.group(2)
            result = self.dispatch(tool_name, args)
            # Keep original call for context, add result after
            return (
                f"<tool>{tool_name}({args})</tool>"
                f"<tool_result>{result}</tool_result>"
            )

        return TOOL_CALL_PATTERN.sub(replace_call, text)


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from model output."""
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else None


def extract_thinking(text: str) -> Optional[str]:
    """Extract the chain-of-thought reasoning from model output."""
    matches = THINK_PATTERN.findall(text)
    return "\n".join(m.strip() for m in matches) if matches else None


def build_system_prompt() -> str:
    """Build the system prompt that teaches the model its tool-use format.

    This prompt is prepended to every conversation. It defines the
    output format and available tools.
    """
    return """You are a helpful reasoning assistant with access to tools.

When solving a problem, follow this format:

<think>
  Your step-by-step reasoning here. Think through the problem carefully.
  Break it into sub-problems. Identify what information you need.
</think>

If you need to use a tool:
<tool>tool_name(arguments)</tool>

Wait for the tool result, then continue reasoning with the new information.

When you have a complete answer:
<answer>Your final, clear answer here.</answer>

Available tools:
- calc(expression): Evaluate math. Examples: calc(15 * 0.20), calc(sqrt(144) + 5**2)
- search(query): Look up factual information. Example: search(population of Switzerland)
- datetime(query): Get date/time info. Examples: datetime(today), datetime(days_until(2025-12-25))

Rules:
1. Always think before acting.
2. Use tools when you need specific facts or calculations.
3. You can use multiple tools across multiple rounds.
4. Always end with <answer>...</answer>.
5. Be concise but complete in your final answer.
"""


class ToolUseInferenceLoop:
    """Multi-step inference loop with tool use.

    Orchestrates the Thought → Action → Observation cycle:

    Round 0: Model sees question → generates <think>...</think> + <tool>...</tool>
    Round 1: Tool result injected → model sees observation → continues reasoning
    ...
    Round N: Model generates <answer>...</answer> → loop ends

    Args:
        model: MiniMoEStudent (or any model with .generate() method).
        tokenizer: Corresponding tokenizer.
        dispatcher: ToolDispatcher instance.
    """

    def __init__(self, model, tokenizer, dispatcher: ToolDispatcher):
        self.model = model
        self.tokenizer = tokenizer
        self.dispatcher = dispatcher
        self.max_tool_rounds = CFG.inference.max_tool_rounds

    def run(self, user_question: str, show_thinking: bool = True) -> dict:
        """Run the full inference loop for a question.

        Args:
            user_question: The user's question.
            show_thinking: Whether to display internal <think> traces.

        Returns:
            dict with:
              "answer": Final answer string
              "thinking": All CoT traces
              "tool_calls": List of tool calls made
              "full_context": Complete context string
              "rounds": Number of tool-use rounds
        """
        import torch

        system_prompt = build_system_prompt()

        # Build initial context in chat format
        # Qwen3 uses the standard chat template with <|im_start|> tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]

        # Convert to model input using chat template
        context = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        console.print(Panel(
            f"[bold]Question:[/] {user_question}",
            title="🤔 Starting Inference Loop",
            border_style="blue"
        ))

        all_generated_text = ""
        answer = None
        rounds = 0

        for round_num in range(self.max_tool_rounds + 1):
            # Tokenize current context
            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=CFG.model.max_seq_len - CFG.inference.max_new_tokens,
            )

            device = next(self.model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Generate next segment
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=CFG.inference.max_new_tokens,
                    temperature=CFG.inference.temperature,
                )

            # Decode only the newly generated tokens
            new_tokens = generated_ids[0, input_ids.shape[1]:]
            new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            all_generated_text += new_text

            # Display thinking if requested
            thinking = extract_thinking(new_text)
            if thinking and show_thinking:
                console.print(Panel(
                    Text(thinking, style="dim italic"),
                    title=f"💭 Thinking (Round {round_num})",
                    border_style="yellow",
                ))

            # Check for final answer
            answer = extract_answer(new_text)
            if answer:
                console.print(Panel(
                    f"[bold green]{answer}[/]",
                    title="✅ Final Answer",
                    border_style="green",
                ))
                break

            # Check for tool calls
            tool_calls = self.dispatcher.parse_tool_calls(new_text)
            if not tool_calls:
                # No tools, no answer → model finished without answer tag
                console.print(
                    "[yellow]⚠ Model stopped without <answer> tag.[/]"
                )
                answer = new_text.strip()
                break

            # Execute tools and inject results
            console.print(
                f"[cyan]🔧 Round {round_num + 1}: "
                f"Executing {len(tool_calls)} tool call(s)...[/]"
            )
            enriched_text = self.dispatcher.inject_results(new_text)

            for call in self.dispatcher.call_history[-len(tool_calls):]:
                console.print(
                    f"  [cyan]→ {call['tool']}({call['args']})[/] "
                    f"= [white]{call['result']}[/]"
                )

            # Append enriched text to context for next round
            context += enriched_text
            rounds += 1

        return {
            "answer": answer or "No answer generated.",
            "thinking": extract_thinking(all_generated_text),
            "tool_calls": self.dispatcher.call_history,
            "full_context": all_generated_text,
            "rounds": rounds,
        }
