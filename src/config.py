"""
config.py — Central configuration for Mini-MoE-CoT pipeline.

All hyperparameters, model choices, and VRAM budgets live here.
Change this file to experiment; no other files need editing for
standard runs.

VRAM budget breakdown (12GB card):
  Base model weights (Qwen3-4B, 4-bit NF4):  ~3.5 GB
  MoE adapter layers (FP16):                 ~0.5 GB
  KV cache (4K context, FP16):               ~0.8 GB
  Gradients + optimizer states (LoRA only):  ~2.5 GB
  Activations + overhead:                    ~1.5 GB
  Safety buffer:                             ~3.2 GB
  ─────────────────────────────────────────────────
  Total estimated peak:                      ~12.0 GB ✓
"""

from pydantic import BaseModel, Field
from typing import Literal


class ModelConfig(BaseModel):
    """Base LLM configuration."""

    # The frozen base model. Qwen3-4B fits 12GB with 4-bit quantization.
    # Alternative: "google/gemma-3-4b-it" (requires HF access token)
    base_model_id: str = "Qwen/Qwen3-4B-Instruct"

    # 4-bit quantization keeps the base model at ~3.5GB VRAM.
    # "nf4" (Normal Float 4) is best quality; "fp4" is slightly faster.
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"

    # Double quantization: quantize the quantization constants too.
    # Saves ~0.1GB with minimal quality loss.
    bnb_4bit_use_double_quant: bool = True

    # Compute dtype for the 4-bit dequantization. bfloat16 is stable
    # on Ampere+ GPUs (RTX 3000+). Use float16 for older cards.
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Maximum sequence length. 4096 is safe for 12GB; reduce to 2048
    # if you see OOM errors during training.
    max_seq_len: int = 4096


class MoEConfig(BaseModel):
    """Mixture-of-Experts layer configuration."""

    # Number of expert FFN networks. 4 is a good start for learning:
    # enough to see routing behavior, small enough to train fast.
    num_experts: int = 4

    # How many experts are active per token (top-k routing).
    # top_k=2 means each token is processed by 2 out of 4 experts,
    # then their outputs are weighted and summed.
    top_k: int = 2

    # Hidden dimension of each expert FFN.
    # Matched to Qwen3-4B's hidden size (2560) for compatibility.
    hidden_size: int = 2560

    # Intermediate size inside each expert (typically 4x hidden_size,
    # but we reduce it since we have multiple experts).
    intermediate_size: int = 5120

    # Expert dropout during training (prevents over-reliance on one expert).
    expert_dropout: float = 0.1

    # Load balancing loss weight (auxiliary loss).
    # Prevents "expert collapse" where the router ignores most experts.
    # Too high → hurts task performance. Too low → collapse.
    # 0.01 is the DeepSeek recommendation.
    aux_loss_coef: float = 0.01


class LoRAConfig(BaseModel):
    """LoRA adapter configuration for parameter-efficient fine-tuning.

    We freeze the 4-bit base model and only train:
    1. LoRA adapters on attention layers
    2. MoE layer weights (router + experts)

    This keeps trainable params to ~50M instead of 4B.
    """

    # LoRA rank. Higher = more capacity but more memory.
    # r=16 is a good balance for reasoning tasks.
    r: int = 16

    # LoRA alpha (scaling factor). Typically 2x rank.
    lora_alpha: int = 32

    # Which attention modules to add LoRA to.
    # q_proj + v_proj is the standard DeepSeek/QLoRA recipe.
    target_modules: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Dropout in LoRA adapters.
    lora_dropout: float = 0.05

    # Whether to add LoRA to the embedding layer.
    bias: str = "none"


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    # Training epochs. 3 epochs on 2000 samples takes ~2-3 hours on 12GB.
    num_epochs: int = 3

    # Batch size. Keep at 1 with gradient accumulation for 12GB safety.
    per_device_train_batch_size: int = 1

    # Simulate larger batch size by accumulating gradients.
    # Effective batch = 1 * 8 = 8.
    gradient_accumulation_steps: int = 8

    # Peak learning rate for cosine schedule.
    learning_rate: float = 2e-4

    # Warmup steps (helps stability at start of training).
    warmup_steps: int = 50

    # Gradient clipping (prevents exploding gradients in LoRA + MoE).
    max_grad_norm: float = 1.0

    # Save checkpoint every N steps.
    save_steps: int = 100

    # Log metrics every N steps.
    logging_steps: int = 10

    # Weight decay for AdamW optimizer.
    weight_decay: float = 0.01

    # Output directory for checkpoints.
    output_dir: str = "checkpoints"

    # TensorBoard log directory.
    logging_dir: str = "logs"

    # Use gradient checkpointing to trade compute for memory.
    # Saves ~2GB VRAM at cost of ~30% slower training. Recommended.
    gradient_checkpointing: bool = True

    # Mixed precision. "bf16" is best for Ampere+ (RTX 3000+).
    # Use "fp16" for older cards, "no" to disable.
    bf16: bool = True
    fp16: bool = False


class DistillConfig(BaseModel):
    """Data distillation from Ollama teacher configuration."""

    # Teacher model name as known to Ollama.
    # Run: `ollama list` to see what you have installed.
    # Recommended: "qwen3.5:27b" (best quality) or "gemma4:27b"
    # Budget option: "qwen3.5:7b" or "gemma4:12b"
    teacher_model: str = "qwen3.5:27b"

    # Ollama server address (default local).
    ollama_host: str = "http://localhost:11434"

    # Number of training examples to generate.
    # 2000 is enough for a meaningful demo. 5000+ for real learning.
    n_samples: int = 2000

    # Output path for the JSONL dataset.
    output_path: str = "data/cot_dataset.jsonl"

    # Temperature for teacher sampling. Higher = more diverse traces.
    temperature: float = 0.7

    # Maximum tokens for teacher response (CoT traces can be long).
    max_tokens: int = 2048

    # Whether to include tool-use examples (vs pure CoT).
    include_tool_examples: bool = True

    # Fraction of examples that include tool calls.
    tool_example_ratio: float = 0.4


class InferenceConfig(BaseModel):
    """Inference-time settings."""

    # Maximum new tokens to generate per step.
    max_new_tokens: int = 512

    # Temperature for student sampling.
    temperature: float = 0.3

    # Maximum tool call rounds before stopping (prevents infinite loops).
    max_tool_rounds: int = 5

    # Whether to show internal <think> traces in output.
    show_thinking: bool = True


class Config(BaseModel):
    """Root config — compose all sub-configs here."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    moe: MoEConfig = Field(default_factory=MoEConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    distill: DistillConfig = Field(default_factory=DistillConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


# ── Singleton: import this anywhere ────────────────────────────────────────
CFG = Config()
