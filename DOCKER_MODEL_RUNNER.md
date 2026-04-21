# Docker Model Runner Setup

Docker Model Runner is Docker Desktop's built-in LLM inference service (available in Docker Desktop 4.40+). It provides an OpenAI-compatible API and can be used as an alternative to Ollama for the teacher model during distillation.

## Why Use Docker Model Runner?

- **Integrated with Docker Desktop** - No separate service to manage
- **OpenAI-compatible API** - Easy to integrate with existing code
- **Quantized models** - Uses efficient Q4_K_M quantization for memory efficiency
- **GPU acceleration** - Leverages your GPU through llama.cpp

## Prerequisites

- Docker Desktop 4.40 or later
- Enable Model Runner in Docker Desktop settings or via CLI:
  ```bash
  docker desktop enable model-runner
  ```

## Quick Start

### 1. Enable Docker Model Runner

```bash
docker desktop enable model-runner --tcp=12434
```

This enables TCP access on port 12434 for host processes. From within Docker containers, use `http://model-runner.docker.internal/engines/v1`.

### 2. Pull a Teacher Model

Browse available models at https://hub.docker.com/u/ai. For teaching, we recommend:

```bash
# Smaller, faster model (good for testing)
docker model pull ai/qwen2.5:7B-Q4_K_M

# Larger, higher quality model (slower)
docker model pull ai/qwen2.5:14B-Q4_K_M
```

### 3. Use with Mini-MoE-CoT

Set the environment variable to use Docker Model Runner instead of Ollama:

```bash
# On Linux/Mac
export TEACHER_BACKEND=docker_model_runner
export TEACHER_MODEL=ai/qwen2.5:7B-Q4_K_M

# On Windows PowerShell
$env:TEACHER_BACKEND = "docker_model_runner"
$env:TEACHER_MODEL = "ai/qwen2.5:7B-Q4_K_M"

# On Windows CMD
set TEACHER_BACKEND=docker_model_runner
set TEACHER_MODEL=ai/qwen2.5:7B-Q4_K_M
```

Then run distillation normally:

```bash
python -m src.distill --n_samples 500
```

The script will automatically use Docker Model Runner instead of Ollama.

## Docker Container Setup

When running Mini-MoE-CoT in a Docker container, you can access the host's Docker Model Runner via the special DNS name:

```bash
# In docker-compose.yml, add this environment variable:
environment:
  - TEACHER_BACKEND=docker_model_runner
  - TEACHER_MODEL=ai/qwen2.5:7B-Q4_K_M
  - DOCKER_MODEL_RUNNER_URL=http://host.docker.internal:12434/engines/v1
```

Then rebuild and run:

```bash
docker-compose down
docker-compose build
docker-compose up -d
docker-compose exec moe-distill python -m src.distill --n_samples 500
```

## Available Models

Check Docker Hub for the latest models: https://hub.docker.com/u/ai

Popular choices for teacher models:
- `ai/qwen2.5:7B-Q4_K_M` - Fast, good quality (~4GB VRAM)
- `ai/qwen2.5:14B-Q4_K_M` - Better quality (~8GB VRAM)
- `ai/llama3.1:8B-Q4_K_M` - Meta's Llama 3.1 (~5GB VRAM)

## Troubleshooting

### Model Runner not accessible

Ensure Docker Model Runner is enabled:
```bash
docker desktop enable model-runner
```

Check if it's running:
```bash
docker model list
```

### Connection refused from container

The container needs to access the host's Model Runner. Use `host.docker.internal` as the hostname:
```bash
export DOCKER_MODEL_RUNNER_URL=http://host.docker.internal:12434/engines/v1
```

### Model not found

Make sure you've pulled the model first:
```bash
docker model pull ai/qwen2.5:7B-Q4_K_M
```

## Comparison: Ollama vs Docker Model Runner

| Feature | Ollama | Docker Model Runner |
|---------|--------|---------------------|
| Setup | Separate installation | Built into Docker Desktop |
| API | Custom API | OpenAI-compatible |
| Model format | GGUF | GGUF (via llama.cpp) |
| Model selection | ollama.com | Docker Hub (ai/ namespace) |
| Container access | Requires host networking | Special DNS name |
| Platform support | Linux/Mac/Windows | Mac (Apple Silicon), Windows (limited) |

**Note:** Docker Model Runner is currently best supported on macOS with Apple Silicon. Windows support is evolving. For Windows, Ollama may be more reliable at this time.
