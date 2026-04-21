"""
setup_windows.py — Windows + CUDA environment setup and validation.

Run this BEFORE anything else to validate your environment:
  python setup_windows.py

What it checks:
  1. Python version (3.10+ required)
  2. CUDA availability and VRAM
  3. PyTorch CUDA build
  4. bitsandbytes Windows compatibility
  5. Ollama connectivity
  6. Disk space for model weights

What it installs / configures:
  - Sets PYTORCH_CUDA_ALLOC_CONF for better memory fragmentation handling
  - Sets BNB_CUDA_VERSION to match your CUDA install
  - Creates .env file with recommended Windows settings
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

# ── ANSI colors (work on Windows 10+ terminal) ──────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def ok(msg): print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")
def rule(): print("─" * 55)


def check_python():
    header("1. Python Version")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} detected — need 3.10+")
        info("Download from: https://python.org/downloads/")
        sys.exit(1)


def check_cuda():
    header("2. CUDA / PyTorch")
    try:
        import torch
        ok(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"CUDA available: {device_name}")
            ok(f"VRAM: {vram_gb:.1f} GB")

            if vram_gb < 10:
                warn(f"Only {vram_gb:.1f} GB VRAM — need 12GB for comfortable training.")
                warn("You can still train with gradient_checkpointing=True and reduced context.")
            elif vram_gb >= 12:
                ok(f"{vram_gb:.1f} GB VRAM — sufficient for this pipeline.")

            cuda_ver = torch.version.cuda
            ok(f"CUDA version: {cuda_ver}")

            return vram_gb
        else:
            fail("CUDA not available! Check NVIDIA driver + PyTorch CUDA build.")
            info("Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")
            info("Choose CUDA 12.1: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return 0.0

    except ImportError:
        fail("PyTorch not installed.")
        info("pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)


def check_bitsandbytes():
    header("3. bitsandbytes (4-bit quantization)")
    try:
        import bitsandbytes as bnb
        ok(f"bitsandbytes {bnb.__version__}")

        # Try a quick 4-bit operation
        try:
            import torch
            if torch.cuda.is_available():
                # Lightweight test: create a 4-bit linear layer
                layer = bnb.nn.Linear4bit(64, 64, bias=False)
                ok("4-bit linear layer creation OK")
        except Exception as e:
            warn(f"4-bit test failed: {e}")
            info("Try: pip install bitsandbytes --upgrade")

    except ImportError:
        fail("bitsandbytes not installed.")
        info("pip install bitsandbytes")
        info("Note: Windows support requires bitsandbytes >= 0.43.0")


def check_transformers():
    header("4. HuggingFace Transformers + PEFT")
    packages = {
        "transformers": "transformers",
        "peft": "peft",
        "accelerate": "accelerate",
        "datasets": "datasets",
    }
    for pkg, import_name in packages.items():
        try:
            mod = __import__(import_name)
            ok(f"{pkg} {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            fail(f"{pkg} not installed — run: pip install {pkg}")


def check_ollama():
    header("5. Ollama (teacher model server)")
    try:
        import ollama
        ok(f"ollama Python client installed")

        # Try to connect
        try:
            client = ollama.Client()
            models = client.list()
            model_names = [m.model for m in models.models]
            ok(f"Ollama server reachable")

            if model_names:
                ok(f"Available models: {', '.join(model_names[:5])}")
                # Check for recommended models
                recommended = ["qwen3.5:27b", "gemma4:27b", "qwen3.5:7b", "gemma4:12b"]
                found = [m for m in recommended if any(m in name for name in model_names)]
                if found:
                    ok(f"Recommended teacher model found: {found[0]}")
                else:
                    warn("No recommended teacher model found.")
                    info("Pull one with: ollama pull qwen3.5:7b  (smaller, faster)")
                    info("Or:           ollama pull qwen3.5:27b (better quality)")
            else:
                warn("No models pulled yet.")
                info("Pull a teacher model: ollama pull qwen3.5:7b")

        except Exception as e:
            warn(f"Ollama server not reachable: {e}")
            info("Start Ollama: Download from https://ollama.com and run 'ollama serve'")

    except ImportError:
        fail("ollama Python package not installed.")
        info("pip install ollama")


def check_disk():
    header("6. Disk Space")
    # Check space in current directory
    total, used, free = shutil.disk_usage(".")
    free_gb = free / 1e9
    info(f"Free disk space: {free_gb:.1f} GB")

    requirements = {
        "Base model (Qwen3-4B, 4-bit)": 3.0,
        "Training dataset (2000 samples)": 0.5,
        "Checkpoints (3 epochs)": 1.0,
        "Teacher model cache": 0.0,  # Stored by Ollama, not here
    }

    total_needed = sum(requirements.values())
    if free_gb >= total_needed:
        ok(f"Sufficient space ({free_gb:.1f} GB free, need ~{total_needed:.1f} GB)")
    else:
        warn(f"Low disk space: {free_gb:.1f} GB free, recommend {total_needed:.1f}+ GB")


def write_env_file(vram_gb: float):
    """Write recommended Windows environment settings."""
    header("7. Writing .env configuration")

    # Determine context length based on VRAM
    if vram_gb >= 16:
        max_seq_len = 4096
    elif vram_gb >= 12:
        max_seq_len = 3072
    else:
        max_seq_len = 2048

    env_content = f"""# Mini-MoE-CoT Environment Settings for Windows + CUDA
# Generated by setup_windows.py
# Source this file or set these in your terminal before running.

# Better CUDA memory allocation (reduces fragmentation on Windows)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Match your CUDA version (check with: nvcc --version)
BNB_CUDA_VERSION=121

# HuggingFace cache (change if you want models stored elsewhere)
# HF_HOME=C:/Users/YourName/.cache/huggingface

# Ollama server URL (default local)
OLLAMA_HOST=http://localhost:11434

# Recommended sequence length for {vram_gb:.0f}GB VRAM
# (Edit src/config.py ModelConfig.max_seq_len to match)
RECOMMENDED_MAX_SEQ_LEN={max_seq_len}

# Set in PowerShell with:
#   $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
# Or add to System Environment Variables in Windows Settings.
"""

    with open(".env", "w") as f:
        f.write(env_content)
    ok(f".env file written (max_seq_len={max_seq_len} for {vram_gb:.0f}GB VRAM)")

    # Print PowerShell commands
    print(f"""
{CYAN}PowerShell environment setup (run in your terminal):{RESET}
  $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
  $env:BNB_CUDA_VERSION = "121"
""")


def run_component_tests():
    """Run the component tests to validate core code."""
    header("8. Component Tests (no GPU required)")
    result = subprocess.run(
        [sys.executable, "test_components.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok("All component tests passed!")
    else:
        warn("Some component tests failed:")
        print(result.stdout[-500:])  # Last 500 chars


def print_next_steps():
    rule()
    print(f"\n{BOLD}Next Steps:{RESET}")
    print(f"""
  {CYAN}1.{RESET} Set environment variables (see PowerShell commands above)

  {CYAN}2.{RESET} Start Ollama with a teacher model:
       ollama run qwen3.5:7b       # Fast, smaller
       ollama run qwen3.5:27b      # Slower, better quality

  {CYAN}3.{RESET} Generate training data:
       python -m src.distill --n_samples 500  # Quick test
       python -m src.distill --n_samples 2000  # Full run

  {CYAN}4.{RESET} Train the student model:
       python -m src.train --data data/cot_dataset.jsonl

  {CYAN}5.{RESET} Run inference:
       python -m src.infer --interactive

  {CYAN}6.{RESET} Visualize expert routing (after training):
       python analyze_routing.py --checkpoint checkpoints/checkpoint-300
       python analyze_routing.py --token-level

{BOLD}TensorBoard (visualize training):{RESET}
  tensorboard --logdir logs
  → Open http://localhost:6006 in browser
""")
    rule()


def main():
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  Mini-MoE-CoT — Windows + CUDA Setup Validator{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")

    check_python()
    vram_gb = check_cuda()
    check_bitsandbytes()
    check_transformers()
    check_ollama()
    check_disk()
    write_env_file(vram_gb)
    run_component_tests()
    print_next_steps()


if __name__ == "__main__":
    main()
