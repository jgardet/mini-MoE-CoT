# Docker Setup for Mini-MoE-CoT

This project can be run entirely in Docker, which is useful when:
- You cannot install Python/CUDA locally
- You need a reproducible environment
- You want to isolate dependencies

## Prerequisites

- Docker installed
- NVIDIA Docker runtime (nvidia-docker2) for GPU support
- NVIDIA GPU with CUDA support (12GB+ VRAM recommended)

**GPU Compatibility Note:**
- RTX 40-series (sm_89): Fully supported
- Blackwell RTX PRO 2000 (sm_120): PyTorch in this Dockerfile does not yet support sm_120. Consider:
  - **⚠️ CPU-only mode (strongly discouraged):** 10-100x slower than GPU, only for testing
  - Waiting for PyTorch stable release with sm_120 support
  - Using PyTorch nightly builds (unstable)
  - Running on RTX 4080 instead where supported

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
```

Or with Docker directly:
```bash
docker build -t moe-distill .
```

### 2. Start the Container

```bash
docker-compose up -d
```

This starts the container in detached mode and keeps it running.

### 3. Run Commands Inside the Container

#### Interactive Shell
```bash
docker-compose exec moe-distill bash
```

Or with Docker:
```bash
docker exec -it moe-distill bash
```

#### Run Setup Validation
```bash
docker-compose exec moe-distill python setup_windows.py
```

#### Generate Training Data
```bash
docker-compose exec moe-distill python -m src.distill --n_samples 2000
```

#### Train the Model
```bash
docker-compose exec moe-distill python -m src.train --data data/cot_dataset.jsonl
```

#### Run Inference
```bash
docker-compose exec moe-distill python -m src.infer --interactive
```

#### Run TensorBoard
```bash
# Option 1: Run from main container
docker-compose exec moe-distill tensorboard --logdir=logs --host=0.0.0.0

# Option 2: Use dedicated TensorBoard service
docker-compose --profile tensorboard up tensorboard
```

Then open http://localhost:6006 in your browser.

## Volume Mounts

The following directories are mounted as volumes:
- `./data` → `/app/data` (datasets)
- `./checkpoints` → `/app/checkpoints` (model checkpoints)
- `./logs` → `/app/logs` (training logs)
- `huggingface_cache` → `/root/.cache/huggingface` (model cache)

This means your data and checkpoints persist outside the container.

## GPU Access

The container uses the NVIDIA Docker runtime for GPU access. Make sure you have:

1. NVIDIA drivers installed on the host
2. nvidia-docker2 installed
3. GPU available: `nvidia-smi` should work on the host

To check GPU access inside the container:
```bash
docker-compose exec moe-distill python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### No GPU Access
If you get CUDA errors:
- Check nvidia-docker is installed: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Verify GPU is available on host: `nvidia-smi`

### Out of Memory
- Reduce batch size in `src/config.py`
- Enable gradient checkpointing (already enabled by default)
- Reduce `max_seq_len` in config

### Permission Issues
If you get permission errors with mounted volumes:
```bash
# On Linux, fix permissions
sudo chown -R $USER:$USER data checkpoints logs
```

## Stopping the Container

```bash
docker-compose down
```

To remove volumes as well:
```bash
docker-compose down -v
```

## Building Without GPU (CPU Only)

If you don't have a GPU or want to test without GPU:

1. Modify the Dockerfile to use a CPU base image:
```dockerfile
FROM python:3.11-slim
# Remove CUDA-specific installations
# Install CPU-only PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

2. Remove the `runtime: nvidia` and GPU reservations from docker-compose.yml

3. Training will be much slower and may require reducing model size.
