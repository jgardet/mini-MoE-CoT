# Environment Configuration Guide

The `.env` file allows you to configure Mini-MoE-CoT for different hardware setups without modifying code.

## Quick Start

1. Copy the example configuration for your hardware:
   ```bash
   # For RTX 40-series (12GB+ VRAM) - GPU mode
   cp .env.example.gpu .env

   # For Blackwell RTX PRO 2000 - CPU mode (PyTorch doesn't support sm_120 yet)
   cp .env.example.blackwell .env

   # For CPU-only mode
   cp .env.example.cpu .env
   ```

2. Edit `.env` if needed for your specific setup

3. Run commands normally - the `.env` file is automatically loaded

## Configuration Options

### Teacher Backend

```bash
# Use Ollama (default)
TEACHER_BACKEND=ollama
TEACHER_MODEL=qwen3.5:27b

# Use Docker Model Runner (Docker Desktop 4.40+)
# From inside container: use model-runner.docker.internal
# From host: use localhost:12434 (if TCP enabled)
TEACHER_BACKEND=docker_model_runner
TEACHER_MODEL=ai/qwen2.5:7B-Q4_K_M
DOCKER_MODEL_RUNNER_URL=http://model-runner.docker.internal/engines/v1
```

### Device Selection

```bash
# GPU mode (RTX 40-series, etc.)
DEVICE=cuda

# CPU mode (for unsupported GPUs or testing)
DEVICE=cpu
```

### Model Configuration

```bash
# Base model
base_model_name=Qwen/Qwen3-4B-Instruct

# For CPU mode, use smaller model:
base_model_name=Qwen/Qwen2.5-0.5B-Instruct

# Sequence length (reduce for CPU or low VRAM)
max_seq_len=4096  # Default for GPU
max_seq_len=1024  # For CPU
```

### MoE Configuration

```bash
# Must match base model's hidden_size
# Qwen3-4B: hidden_size=2560
# Qwen2.5-0.5B: hidden_size=896
hidden_size=2560
intermediate_size=5120
num_experts=4
top_k=2
```

### Training Configuration

```bash
# Batch size
batch_size=1  # GPU
batch_size=2  # CPU (smaller batches for memory)

# Epochs
epochs=3

# Learning rate
learning_rate=2e-4

# Gradient checkpointing (saves memory, slower)
gradient_checkpointing=true
```

### Distillation Configuration

```bash
# Number of samples to generate
distill_n_samples=2000

# Teacher sampling parameters
distill_temperature=0.7
distill_max_tokens=2048
```

## Hardware-Specific Configurations

### RTX 4080/4090 (12GB+ VRAM) - Full GPU Mode

```bash
DEVICE=cuda
base_model_name=Qwen/Qwen3-4B-Instruct
max_seq_len=4096
hidden_size=2560
batch_size=1
epochs=3
```

### RTX PRO 2000 Blackwell (sm_120) - CPU Mode

Currently, PyTorch stable builds don't support sm_120. Use CPU mode:

```bash
DEVICE=cpu
base_model_name=Qwen/Qwen2.5-0.5B-Instruct
max_seq_len=1024
hidden_size=896
batch_size=2
epochs=1
TEACHER_BACKEND=docker_model_runner
TEACHER_MODEL=ai/qwen2.5:7B-Q4_K_M
```

### CPU-Only Mode

```bash
DEVICE=cpu
base_model_name=Qwen/Qwen2.5-0.5B-Instruct
max_seq_len=512
hidden_size=896
batch_size=1
epochs=1
```

## Docker Setup with .env

When using Docker, the `.env` file is automatically loaded if you mount it:

```yaml
# docker-compose.yml
services:
  moe-distill:
    env_file:
      - .env
```

Or pass environment variables directly:

```bash
docker-compose run -e DEVICE=cpu -e TEACHER_BACKEND=docker_model_runner moe-distill
```

## Troubleshooting

### .env file not loading

Ensure the `.env` file is in the project root (same directory as `src/`).

### Configuration not taking effect

Check that environment variable names match exactly (case-sensitive).

### GPU not detected despite DEVICE=cuda

Verify PyTorch CUDA installation and GPU compatibility. See GPU Compatibility Notes in README.md.
