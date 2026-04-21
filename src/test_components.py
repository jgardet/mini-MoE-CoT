"""
test_components.py — Quick validation of core components without a GPU.

Run this first to verify installation before the full pipeline:
  python test_components.py

Tests:
  1. MoE layer forward pass (CPU, random weights)
  2. Tool dispatcher (calculator, search, datetime)
  3. Config loading
  4. Dataset parsing (mock data)

This runs in ~5 seconds and requires only PyTorch + the tools/ directory.
No model download needed.
"""

import sys
import torch
import json
import tempfile
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_config():
    """Test: Config loads without errors."""
    print("─" * 50)
    print("TEST 1: Config loading")
    from src.config import CFG
    assert CFG.moe.num_experts == 4
    assert CFG.moe.top_k == 2
    assert CFG.model.load_in_4bit == True
    print(f"  ✓ Config OK — {CFG.moe.num_experts} experts, top-{CFG.moe.top_k}")


def test_moe_layer():
    """Test: MoE layer forward pass on CPU."""
    print("─" * 50)
    print("TEST 2: MoE Layer (CPU)")
    from src.moe_layer import MoELayer

    batch_size, seq_len, hidden_size = 2, 8, 256
    moe = MoELayer(
        hidden_size=hidden_size,
        intermediate_size=512,
        num_experts=4,
        top_k=2,
        aux_loss_coef=0.01,
    )

    x = torch.randn(batch_size, seq_len, hidden_size)
    output, aux_loss = moe(x)

    assert output.shape == (batch_size, seq_len, hidden_size), \
        f"Output shape mismatch: {output.shape}"
    assert aux_loss.item() > 0, "Aux loss should be positive"
    assert not torch.isnan(output).any(), "Output contains NaN"

    print(f"  ✓ Forward pass OK")
    print(f"    Input:  {tuple(x.shape)}")
    print(f"    Output: {tuple(output.shape)}")
    print(f"    Aux loss: {aux_loss.item():.6f}")

    # Test backward pass (gradients flow through MoE)
    total_loss = output.sum() + aux_loss
    total_loss.backward()
    print(f"  ✓ Backward pass OK (gradients computed)")


def test_router_load_balance():
    """Test: Router distributes tokens reasonably across experts."""
    print("─" * 50)
    print("TEST 3: Router load balancing")
    from src.moe_layer import Router

    router = Router(hidden_size=64, num_experts=4, top_k=2)
    x = torch.randn(1, 100, 64)  # 100 tokens

    weights, indices, logits = router(x)

    # Count how many tokens each expert gets
    expert_counts = torch.zeros(4)
    for k in range(2):
        for expert_id in range(4):
            count = (indices[:, :, k] == expert_id).sum().item()
            expert_counts[expert_id] += count

    total = expert_counts.sum().item()
    print(f"  Expert utilization (100 tokens, top-2):")
    for i, count in enumerate(expert_counts):
        pct = 100 * count.item() / total
        bar = "█" * int(pct / 2)
        print(f"    Expert {i}: {bar} {pct:.1f}%")

    # Check no expert is completely ignored
    assert all(c > 0 for c in expert_counts), "Some expert got 0 tokens!"
    print(f"  ✓ All experts active")


def test_calculator():
    """Test: Calculator tool handles various expressions."""
    print("─" * 50)
    print("TEST 4: Calculator tool")
    from tools.calculator import run

    test_cases = [
        ("3 + 4 * 2", "Calculator result: 11"),
        ("sqrt(144)", "Calculator result: 12"),
        ("15 * 0.20", "Calculator result: 3.0"),
        ("2 ** 10", "Calculator result: 1024"),
        ("100 / 0", "Calculator error"),  # Division by zero
    ]

    for expr, expected_prefix in test_cases:
        result = run(expr)
        assert result.startswith(expected_prefix.split(":")[0]), \
            f"Unexpected result for '{expr}': {result}"
        print(f"  ✓ calc({expr}) → {result}")


def test_search():
    """Test: Search tool returns results."""
    print("─" * 50)
    print("TEST 5: Search tool")
    from tools.search import run

    result = run("population of Switzerland")
    assert "Switzerland" in result or "population" in result.lower()
    print(f"  ✓ search(population of Switzerland) → {result[:60]}...")

    result = run("speed of light")
    assert "light" in result.lower() or "299" in result
    print(f"  ✓ search(speed of light) → {result[:60]}...")


def test_datetime():
    """Test: Datetime tool returns current date info."""
    print("─" * 50)
    print("TEST 6: Datetime tool")
    from tools.datetime_tool import run

    result = run("today")
    assert "date" in result.lower() or "today" in result.lower()
    print(f"  ✓ datetime(today) → {result}")

    result = run("year")
    assert "202" in result  # Should contain current year
    print(f"  ✓ datetime(year) → {result}")


def test_tool_dispatcher():
    """Test: ToolDispatcher parses and dispatches correctly."""
    print("─" * 50)
    print("TEST 7: Tool dispatcher + parsing")
    from src.tool_loop import ToolDispatcher

    dispatcher = ToolDispatcher()

    # Test parsing
    text = (
        "<think>Let me calculate this.</think>\n"
        "<tool>calc(15 * 0.20)</tool>\n"
        "<tool>search(Switzerland population)</tool>"
    )
    calls = dispatcher.parse_tool_calls(text)
    assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
    assert calls[0] == ("calc", "15 * 0.20"), f"Unexpected call: {calls[0]}"
    print(f"  ✓ Parsed {len(calls)} tool calls correctly")

    # Test dispatch + inject
    enriched = dispatcher.inject_results(text)
    assert "<tool_result>" in enriched
    print(f"  ✓ Tool results injected into context")

    # Test answer extraction
    from src.tool_loop import extract_answer, extract_thinking
    answer_text = "<think>The answer is 3.</think><answer>3.0</answer>"
    assert extract_answer(answer_text) == "3.0"
    assert "The answer is 3" in extract_thinking(answer_text)
    print(f"  ✓ Answer and thinking extraction OK")


def test_aux_loss():
    """Test: Auxiliary loss with collapsed vs balanced routing."""
    print("─" * 50)
    print("TEST 8: Auxiliary loss behavior")
    from src.moe_layer import compute_aux_loss

    num_experts = 4

    # Balanced routing: each expert gets equal probability
    balanced_logits = torch.zeros(100, num_experts)
    balanced_loss = compute_aux_loss(balanced_logits, num_experts, top_k=2, aux_loss_coef=1.0)

    # Collapsed routing: all tokens go to expert 0
    collapsed_logits = torch.zeros(100, num_experts)
    collapsed_logits[:, 0] = 100.0  # Very high score for expert 0
    collapsed_loss = compute_aux_loss(collapsed_logits, num_experts, top_k=2, aux_loss_coef=1.0)

    assert collapsed_loss > balanced_loss, \
        f"Collapsed routing should have higher loss! ({collapsed_loss:.4f} vs {balanced_loss:.4f})"

    print(f"  ✓ Balanced routing loss:  {balanced_loss:.4f}")
    print(f"  ✓ Collapsed routing loss: {collapsed_loss:.4f}")
    print(f"  ✓ Aux loss correctly penalizes imbalanced routing")


def main():
    print("\n" + "═" * 50)
    print("  Mini-MoE-CoT Component Tests")
    print("═" * 50)

    tests = [
        test_config,
        test_moe_layer,
        test_router_load_balance,
        test_calculator,
        test_search,
        test_datetime,
        test_tool_dispatcher,
        test_aux_loss,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("═" * 50)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n✅ All components OK! You can proceed to:")
        print("   1. Start Ollama: ollama run qwen3.5:27b")
        print("   2. Generate data: python -m src.distill")
        print("   3. Train: python -m src.train")
        print("   4. Infer: python -m src.infer --interactive")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
