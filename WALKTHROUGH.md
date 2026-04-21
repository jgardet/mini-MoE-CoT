# Interactive Walkthrough Notebook

The `src/walkthrough.ipynb` notebook provides a hands-on, interactive learning experience for understanding each component of the Mini-MoE-CoT pipeline.

## Overview

Run the notebook cells in order to explore:
1. **MoE Routing** — How the router decides which experts to use
2. **Expert Specialization** — Why experts diverge over training
3. **Chain-of-Thought Format** — How traces are structured
4. **Tool Use Pattern** — The ReAct loop: Think → Act → Observe
5. **Distillation** — Calling the Ollama teacher (requires Ollama)
6. **Training Loop** — Mini training run on toy data (requires GPU)

## Requirements

- **Sections 1-4**: No GPU required, just Python with PyTorch
- **Section 5**: Requires Ollama running locally with a model (e.g., `ollama run qwen3.5:7b`)
- **Section 6**: Requires GPU for actual training (simulated without GPU)

## Quick Start

```bash
# Install Jupyter if not already installed
pip install jupyter

# Navigate to project directory
cd c:\Work\research\Mini-MoE-CoT

# Start Jupyter
jupyter notebook

# Open src/walkthrough.ipynb in your browser
```

## What You'll Learn

| Component | Key Insight |
|-----------|-------------|
| **MoE Router** | Softmax over N experts → top-k selection. Differentiable. |
| **Aux Loss** | Prevents collapse by penalizing uneven expert usage. |
| **CoT Format** | traces are supervised: student imitates teacher reasoning. |
| **Tool Loop** | ReAct pattern: generate → parse tags → execute → inject → repeat. |
| **Distillation** | Sequence-level KD: train student on teacher's output tokens. |
| **Specialization** | Emergent behavior: experts develop distinct roles during training. |

## Notebook Sections

### Part 1: MoE Routing
Visualize how the router assigns tokens to experts. Compare balanced vs. collapsed routing with auxiliary loss.

### Part 2: Full MoE Forward Pass
Run the complete MoE layer, verify tensor shapes, gradients, and parameter counts.

### Part 3: Chain-of-Thought Format
Explore the structured XML-like tags used for CoT supervision. Extract thinking and answer components.

### Part 4: Tool Use Execution Loop
Test each available tool (calculator, search, datetime) and simulate the full ReAct loop.

### Part 5: Distillation (requires Ollama)
Generate examples using the teacher model and validate the response format.

### Part 6: Expert Specialization
Visualize how routing distributions change from uniform (before training) to specialized (after training).

## Connection to Production Systems

The notebook also connects concepts to real-world applications:

- **Controlled autonomy**: The `max_tool_rounds` limit is a governance mechanism that scales to enterprise policy engines
- **Observability**: The `call_history` provides an audit trail for regulated environments
- **Expert routing as policy**: MoE router specialization maps to domain-specific sub-agents (financial, legal, etc.)

## Tips

- Run cells sequentially — later cells depend on earlier imports
- The notebook saves visualization outputs as PNG files
- If Ollama is unavailable, Section 5 will skip gracefully
- GPU sections can be run on CPU for testing (slower)
