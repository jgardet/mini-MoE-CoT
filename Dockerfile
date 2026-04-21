# Use NVIDIA CUDA base image with Python 3.11
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV BNB_CUDA_VERSION=121

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt /tmp/

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install -r /tmp/requirements.txt

# Install flash-attn (optional, for faster attention on Linux)
# RUN pip install flash-attn --no-build-isolation

# Set working directory
WORKDIR /app

# Copy project files
COPY src/ /app/src/
COPY setup_windows.py /app/
COPY README.md /app/
COPY requirements.txt /app/

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs

# Expose ports for TensorBoard (6006) and Ollama (11434 if running locally)
EXPOSE 6006 11434

# Default command
CMD ["python", "-m", "src.train", "--help"]
