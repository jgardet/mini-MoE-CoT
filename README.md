# Mini-MoE-CoT: A Teaching Pipeline for Mixture-of-Experts + Chain-of-Thought + Tool Use

A small-scale, fully documented pipeline for learning how modern reasoning models
are built. Distills from a local teacher (Ollama) into a tiny student MoE, with
Chain-of-Thought supervision and multi-step tool use.

## Architecture Overview

```
PHASE 1: DATA DISTILLATION
─────────────────────────────────────────────────────
Teacher (Ollama) → Synthetic Dataset
Gemma4:27b or Qwen3.5:27b
    ↓
CoT traces + tool call sequences → JSONL format

─────────────────────────────────────────────────────

PHASE 2: STUDENT MODEL (Mini-MoE)
─────────────────────────────────────────────────────
Base: Qwen3-4B (4-bit quantized, ~5GB VRAM)
    ↓
MoE FFN Layer (on frozen base):
    Router (learned) → Expert Pool (N=4, top-k=2)
    ├─ Expert 0: Reasoning / Math
    ├─ Expert 1: Tool Planning
    ├─ Expert 2: Synthesis / Summary
    └─ Expert 3: World Knowledge
    ↓
CoT Head (generates reasoning before answer)

─────────────────────────────────────────────────────

PHASE 3: TOOL USE LOOP (inference)
─────────────────────────────────────────────────────
Student generates → <tool>calc(3+4)</tool>
    ↓
ToolDispatcher executes → returns result
    ↓
Result injected → Student continues reasoning
    ↓
Loop until <answer>...</answer> emitted
```

## Key Concepts Demonstrated

| Concept           | Where in code          | Description                                   |
|-------------------|------------------------|-----------------------------------------------|
| MoE routing       | `src/moe_layer.py`     | Softmax router + top-k expert selection       |
| Load balancing    | `src/moe_layer.py`     | Auxiliary loss prevents expert collapse       |
| CoT distillation  | `src/distill.py`       | Teacher generates `<think>` traces            |
| SFT training      | `src/train.py`         | Supervised fine-tuning on distilled data      |
| Tool use          | `src/tool_loop.py`     | Regex-based tool call parsing + dispatch      |
| VRAM budgeting    | `src/model.py`         | 4-bit quantized base + FP16 adapters          |

## Hardware Requirements (Windows + CUDA)

- GPU: 12GB VRAM (RTX 3080/4070/etc.)
- RAM: 16GB+ recommended
- Disk: ~15GB (model weights + dataset)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start teacher model in Ollama (separate terminal)
ollama run qwen3.5:27b   # or: ollama run gemma4:27b

# 3. Generate distillation dataset (~1-2 hours for 2000 examples)
python -m src.distill --n_samples 2000 --output data/cot_dataset.jsonl

# 4. Train the student MoE
python -m src.train --data data/cot_dataset.jsonl --epochs 3

# 5. Run inference with tool loop
python -m src.infer --prompt "What is 15% of 847, and is that more than the square root of 100?"
```

## Docker Setup

For containerized deployment with GPU support, see [DOCKER.md](DOCKER.md) for complete setup instructions and troubleshooting.

## Teacher Model Options

For distillation, you can use either:
- **Ollama** (default) - Local LLM server, see Quick Start above
- **Docker Model Runner** - Built into Docker Desktop 4.40+, see [DOCKER_MODEL_RUNNER.md](DOCKER_MODEL_RUNNER.md)

Set via environment variable:
```bash
export TEACHER_BACKEND=docker_model_runner  # or "ollama" (default)
```

## GPU Compatibility Notes

- **RTX 40-series (sm_89)**: Fully supported with current PyTorch/CUDA
- **Blackwell RTX PRO 2000 (sm_120)**: PyTorch stable builds do not yet support sm_120. Options:
  - Use CPU-only mode (slower)
  - Wait for PyTorch to add sm_120 support (check PyTorch GitHub issues)
  - Use PyTorch nightly builds (unstable)
  - On RTX 4080: Current setup works perfectly

## Interactive Walkthrough

For hands-on learning with the interactive notebook, see [WALKTHROUGH.md](WALKTHROUGH.md). The notebook covers MoE routing, expert specialization, CoT format, tool use, distillation, and training — with visualizations and code examples you can run yourself.

## File Structure

```
mini_moe_cot/
├── README.md               ← You are here
├── DOCKER.md               ← Docker setup instructions
├── DOCKER_MODEL_RUNNER.md  ← Docker Model Runner teacher setup
├── WALKTHROUGH.md          ← Interactive notebook guide
├── requirements.txt        ← All dependencies
├── Dockerfile              ← Container image definition
├── docker-compose.yml      ← Container orchestration
├── setup_windows.py        ← Windows setup validation
├── src/
│   ├── config.py           ← Central config (VRAM budgets, hyperparams)
│   ├── moe_layer.py        ← MoE implementation (router + experts)
│   ├── model.py            ← Full model: base LLM + MoE head
│   ├── distill.py          ← Data generation via teacher (Ollama/Docker Model Runner)
│   ├── dataset.py          ← Dataset loading + tokenization
│   ├── train.py            ← Training loop with CoT loss
│   ├── tool_loop.py        ← Tool dispatcher + inference loop
│   ├── infer.py            ← CLI entry point
│   └── walkthrough.ipynb   ← Interactive learning notebook
├── src/tools/
│   ├── calculator.py       ← Math tool (safe eval)
│   ├── search.py           ← Simulated search tool
│   └── datetime_tool.py    ← Date/time tool
├── data/                   ← Generated JSONL datasets
├── checkpoints/            ← Saved model weights
└── logs/                   ← Training logs (TensorBoard)
```
