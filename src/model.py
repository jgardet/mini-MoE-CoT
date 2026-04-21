"""
model.py — Student model: quantized base LLM + MoE adapter head.

Architecture:
  ┌─────────────────────────────────────────────┐
  │  Frozen 4-bit Base (Qwen3-4B)               │
  │  + LoRA adapters on attention (trainable)   │
  │  → last hidden state                        │
  └───────────────────┬─────────────────────────┘
                      │ hidden_states (B, T, 2560)
                      ▼
  ┌─────────────────────────────────────────────┐
  │  MoE Layer (trainable)                      │
  │  Router + 4 Expert FFNs (top-2 active)      │
  └───────────────────┬─────────────────────────┘
                      │ enriched hidden_states
                      ▼
  ┌─────────────────────────────────────────────┐
  │  LM Head (tied to base model vocab)         │
  │  Projects to vocab logits for generation    │
  └─────────────────────────────────────────────┘

Why this design?
  The base model's LM head already knows how to map hidden states →
  token probabilities. We insert the MoE layer *before* the LM head
  to enrich the representations with specialized expert processing,
  without retraining the vocabulary projection.

  This is sometimes called "model surgery" or "head adapter" pattern.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from .moe_layer import MoELayer
from .config import CFG
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_base_model_and_tokenizer():
    """Load the quantized base model and tokenizer.

    Steps:
    1. Configure 4-bit quantization (NF4 + double quant = ~3.5GB VRAM)
    2. Load model with device_map="auto" (uses GPU first, spills to CPU if needed)
    3. Prepare for k-bit training (fixes gradient issues with quantized layers)
    4. Apply LoRA adapters to attention layers

    Returns:
        model: PEFT-wrapped base model (LoRA adapters trainable, base frozen)
        tokenizer: Corresponding tokenizer
    """
    quant_status = "4-bit NF4 + double quant" if CFG.model.load_in_4bit else "None (full precision)"
    vram_estimate = "~3.5 GB" if CFG.model.load_in_4bit else "~2 GB (CPU mode)"
    console.print(Panel(
        f"[bold cyan]Loading base model:[/] {CFG.model.base_model_id}\n"
        f"[bold cyan]Quantization:[/] {quant_status}\n"
        f"[bold cyan]Estimated VRAM:[/] {vram_estimate}",
        title="📦 Model Loading",
        border_style="cyan"
    ))

    # --- 4-bit quantization config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CFG.model.load_in_4bit,
        bnb_4bit_quant_type=CFG.model.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=CFG.model.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, CFG.model.bnb_4bit_compute_dtype),
    )

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        CFG.model.base_model_id,
        trust_remote_code=True,
        padding_side="right",   # Right padding for training (CausalLM)
    )
    # Ensure pad token exists (some models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    # Use float32 on CPU, bfloat16 on CUDA
    model_dtype = torch.float32 if CFG.model.device == "cpu" else getattr(torch, CFG.model.bnb_4bit_compute_dtype)
    
    # Only pass quantization_config if actually using 4-bit
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
    }
    
    # Set device_map based on mode
    if CFG.model.device == "cpu":
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = "auto"
    
    if CFG.model.load_in_4bit:
        load_kwargs["quantization_config"] = bnb_config
    
    base_model = AutoModelForCausalLM.from_pretrained(
        CFG.model.base_model_id,
        **load_kwargs
    )

    # Required before applying LoRA to a quantized model.
    # This casts layer norms to fp32 and enables gradient checkpointing properly.
    # Only needed when using quantization.
    if CFG.model.load_in_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=CFG.training.gradient_checkpointing,
        )

    # --- Apply LoRA adapters ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CFG.lora.r,
        lora_alpha=CFG.lora.lora_alpha,
        target_modules=CFG.lora.target_modules,
        lora_dropout=CFG.lora.lora_dropout,
        bias=CFG.lora.bias,
        # Only train the LoRA adapters, not the base weights
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    console.print(
        f"[green]✓ LoRA applied[/] — "
        f"Trainable: {trainable:,} params ({100*trainable/total:.1f}% of total)"
    )

    return model, tokenizer


class MiniMoEStudent(nn.Module):
    """Full student model: base LLM + MoE head.

    The MoE layer is inserted between the final transformer hidden state
    and the language model head. During training, only the MoE layer and
    LoRA adapters are updated; the base model weights are frozen.

    Args:
        base_model: PEFT-wrapped quantized base LLM.
        tokenizer: Corresponding tokenizer.
    """

    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

        # The MoE layer is fully trainable
        # It matches the base model's hidden size
        # Use float32 on CPU, bfloat16 on CUDA
        moe_dtype = torch.float32 if CFG.model.device == "cpu" else torch.bfloat16
        self.moe_layer = MoELayer(
            hidden_size=CFG.moe.hidden_size,
            intermediate_size=CFG.moe.intermediate_size,
            num_experts=CFG.moe.num_experts,
            top_k=CFG.moe.top_k,
            aux_loss_coef=CFG.moe.aux_loss_coef,
            expert_dropout=CFG.moe.expert_dropout,
        ).to(dtype=moe_dtype)

        # Move MoE to the same device as the base model
        device = next(base_model.parameters()).device
        self.moe_layer = self.moe_layer.to(device)

        console.print(
            f"[green]✓ MoE layer added[/] — "
            f"{CFG.moe.num_experts} experts, top-{CFG.moe.top_k} routing\n"
            f"  MoE params: {sum(p.numel() for p in self.moe_layer.parameters()):,}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass with MoE augmentation.

        Args:
            input_ids: (batch, seq_len) token ids.
            attention_mask: (batch, seq_len) 1/0 mask.
            labels: (batch, seq_len) target ids for LM loss. Positions with
                    -100 are ignored in the loss (standard HuggingFace convention).

        Returns:
            dict with:
              "loss": total loss (lm_loss + aux_loss) if labels provided
              "logits": (batch, seq_len, vocab_size)
              "aux_loss": load-balancing loss scalar
        """
        # --- Pass through base model, get hidden states ---
        # output_hidden_states=True returns all layer hidden states
        base_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Last layer hidden states: (B, T, hidden_size)
        hidden_states = base_output.hidden_states[-1]

        # --- MoE enrichment ---
        # The MoE layer processes the hidden states with specialized experts
        enriched_states, aux_loss = self.moe_layer(hidden_states)

        # --- Project to vocabulary logits ---
        # Access the base model's LM head (works for PEFT-wrapped models)
        lm_head = self._get_lm_head()
        logits = lm_head(enriched_states.to(lm_head.weight.dtype))

        # --- Compute language modeling loss if labels provided ---
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM loss
            # (predict token i+1 from token i)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Cross-entropy loss (ignoring -100 positions)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Total loss = language modeling + load balancing
            loss = lm_loss + aux_loss

        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": aux_loss,
        }

    def _get_lm_head(self) -> nn.Linear:
        """Retrieve the LM head from the (possibly PEFT-wrapped) model."""
        # PEFT wraps the model; navigate to the underlying base
        model = self.base_model
        while hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "lm_head"):
            return model.lm_head
        elif hasattr(model, "embed_out"):
            return model.embed_out
        else:
            raise AttributeError(
                "Cannot find LM head. Check model architecture."
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Greedy/sampled generation with MoE head.

        This is a simple token-by-token generation loop that:
        1. Runs the full forward pass (base + MoE)
        2. Samples next token from the output logits
        3. Appends to input and repeats

        For production, prefer using the base model's built-in `.generate()`
        with a custom logits processor. This version is explicit for learning.

        Args:
            input_ids: (1, seq_len) input token ids.
            attention_mask: (1, seq_len) attention mask.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            do_sample: If False, use greedy decoding (always pick argmax).

        Returns:
            generated_ids: (1, seq_len + num_generated) full sequence.
        """
        self.eval()
        generated = input_ids.clone()
        mask = attention_mask.clone() if attention_mask is not None else None

        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            out = self.forward(generated, mask)
            next_token_logits = out["logits"][:, -1, :]  # (1, vocab_size)

            if do_sample and temperature > 0:
                # Temperature scaling + categorical sampling
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy: pick highest probability token
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if mask is not None:
                mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)

            # Stop at EOS token
            if next_token.item() == eos_token_id:
                break

        return generated

    def get_trainable_params(self) -> dict:
        """Return a breakdown of trainable vs frozen parameters."""
        groups = {
            "lora_adapters": [],
            "moe_layer": [],
            "frozen_base": [],
        }
        for name, param in self.named_parameters():
            if "moe_layer" in name and param.requires_grad:
                groups["moe_layer"].append(param)
            elif param.requires_grad:
                groups["lora_adapters"].append(param)
            else:
                groups["frozen_base"].append(param)
        return groups


def build_model() -> tuple:
    """Convenience function: load everything and return student model + tokenizer."""
    base_model, tokenizer = load_base_model_and_tokenizer()
    student = MiniMoEStudent(base_model, tokenizer)
    return student, tokenizer
