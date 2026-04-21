"""
train.py — Training loop for the Mini-MoE-CoT student model.

Training strategy:
  1. Only MoE layer + LoRA adapters are trained (base model frozen)
  2. Total loss = LM cross-entropy + auxiliary MoE load-balancing loss
  3. Gradient accumulation simulates larger batch sizes
  4. Cosine LR schedule with linear warmup
  5. TensorBoard logging + periodic checkpointing

Loss decomposition:
  loss_total = loss_lm + α * loss_aux
  
  loss_lm:  Cross-entropy over response tokens. The core "imitate the teacher"
            objective. Minimizing this makes the student generate text similar
            to the teacher's CoT traces and answers.
            
  loss_aux: MoE load-balancing loss. Prevents expert collapse (where the router
            learns to always use the same 1-2 experts). Adds a small penalty
            when expert utilization is uneven.
            α = aux_loss_coef (default 0.01 from config)

Optimizer:
  AdamW on MoE + LoRA parameters only. This is correct: the 4-bit base model
  weights are frozen and don't participate in optimization.
  
  Different learning rates for MoE vs LoRA:
  - MoE: 2e-4 (learning from scratch)
  - LoRA: 1e-4 (fine-tuning pretrained attention)
"""

import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import CFG
from .model import build_model
from .dataset import build_dataloaders

console = Console()


def build_optimizer(model) -> AdamW:
    """Build AdamW optimizer with parameter group-specific LRs.

    We use different learning rates for:
    - MoE layer parameters: trained from scratch, needs higher LR
    - LoRA adapter parameters: fine-tuning, needs lower LR

    This is optional but improves convergence stability.

    Args:
        model: MiniMoEStudent instance.

    Returns:
        Configured AdamW optimizer.
    """
    param_groups = model.get_trainable_params()

    # Verify we have trainable params
    moe_params = param_groups["moe_layer"]
    lora_params = param_groups["lora_adapters"]

    console.print(
        f"[cyan]Optimizer setup:[/]\n"
        f"  MoE params:  {sum(p.numel() for p in moe_params):,} "
        f"(lr={CFG.training.learning_rate})\n"
        f"  LoRA params: {sum(p.numel() for p in lora_params):,} "
        f"(lr={CFG.training.learning_rate * 0.5})"
    )

    optimizer = AdamW(
        [
            {
                "params": moe_params,
                "lr": CFG.training.learning_rate,
                "weight_decay": CFG.training.weight_decay,
            },
            {
                "params": lora_params,
                "lr": CFG.training.learning_rate * 0.5,
                "weight_decay": CFG.training.weight_decay,
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer


def build_scheduler(optimizer: AdamW, total_steps: int):
    """Build a warmup + cosine decay learning rate schedule.

    Schedule:
      Steps 0 → warmup_steps:   Linear warmup from 0 → peak LR
      Steps warmup_steps → end: Cosine decay from peak LR → ~0

    This avoids the large gradient updates at initialization that
    can destabilize 4-bit quantized models.

    Args:
        optimizer: AdamW optimizer.
        total_steps: Total number of gradient update steps.

    Returns:
        SequentialLR scheduler (warmup → cosine).
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=CFG.training.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - CFG.training.warmup_steps,
        eta_min=CFG.training.learning_rate * 0.01,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[CFG.training.warmup_steps],
    )


def train_epoch(
    model,
    dataloader,
    optimizer: AdamW,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    writer: SummaryWriter,
    epoch: int,
    global_step: int,
    device: torch.device,
) -> tuple[float, float, int]:
    """Run one full training epoch.

    Args:
        model: MiniMoEStudent.
        dataloader: Training DataLoader.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler.
        scaler: AMP gradient scaler.
        writer: TensorBoard SummaryWriter.
        epoch: Current epoch number (for logging).
        global_step: Global step counter (for logging/checkpointing).
        device: CUDA device.

    Returns:
        (mean_lm_loss, mean_aux_loss, final_global_step)
    """
    model.train()
    model.base_model.train()

    total_lm_loss = 0.0
    total_aux_loss = 0.0
    n_batches = 0

    # Gradient accumulation: we accumulate over N micro-steps
    accumulation_steps = CFG.training.gradient_accumulation_steps
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # AMP (Automatic Mixed Precision): computes in bf16, accumulates in fp32
        # This saves ~40% memory vs full fp32 training.
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output["loss"]
            aux_loss = output["aux_loss"]

            # Scale loss for gradient accumulation
            # (gradients are summed, so we divide to get the mean)
            loss_scaled = loss / accumulation_steps

        # Backward pass with gradient scaling (for numerical stability with bf16)
        scaler.scale(loss_scaled).backward()

        total_lm_loss += (loss.item() - aux_loss.item())
        total_aux_loss += aux_loss.item()
        n_batches += 1

        # Perform optimizer step every `accumulation_steps` micro-batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Gradient clipping: prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                CFG.training.max_grad_norm,
            )

            # Optimizer step + scheduler step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # TensorBoard logging
            if global_step % CFG.training.logging_steps == 0:
                lm_loss_avg = total_lm_loss / n_batches
                aux_loss_avg = total_aux_loss / n_batches
                lr = scheduler.get_last_lr()[0]

                writer.add_scalar("train/lm_loss", lm_loss_avg, global_step)
                writer.add_scalar("train/aux_loss", aux_loss_avg, global_step)
                writer.add_scalar("train/total_loss", lm_loss_avg + aux_loss_avg, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar(
                    "train/perplexity",
                    math.exp(min(lm_loss_avg, 10)),  # cap to avoid overflow
                    global_step,
                )

                pbar.set_postfix({
                    "lm_loss": f"{lm_loss_avg:.4f}",
                    "aux_loss": f"{aux_loss_avg:.4f}",
                    "lr": f"{lr:.2e}",
                })

    return total_lm_loss / n_batches, total_aux_loss / n_batches, global_step


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device) -> dict:
    """Evaluate the model on the eval set.

    Args:
        model: MiniMoEStudent.
        dataloader: Eval DataLoader.
        device: CUDA device.

    Returns:
        dict with eval metrics.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += output["loss"].item()
            n_batches += 1

    mean_loss = total_loss / n_batches
    perplexity = math.exp(min(mean_loss, 10))

    return {
        "eval_loss": mean_loss,
        "eval_perplexity": perplexity,
    }


def save_checkpoint(model, optimizer, scheduler, epoch: int, step: int, metrics: dict):
    """Save a training checkpoint.

    Saves:
    - MoE layer state dict (full weights)
    - LoRA adapter weights (via PEFT save_pretrained)
    - Optimizer + scheduler state (for resuming training)
    - Training metadata

    Args:
        model: MiniMoEStudent.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler.
        epoch: Current epoch.
        step: Current global step.
        metrics: Dict of eval metrics to save.
    """
    ckpt_dir = Path(CFG.training.output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save MoE layer (custom layer, not part of PEFT)
    torch.save(
        model.moe_layer.state_dict(),
        ckpt_dir / "moe_layer.pt"
    )

    # Save LoRA adapters via PEFT
    model.base_model.save_pretrained(ckpt_dir / "lora_adapters")

    # Save optimizer + scheduler state for resuming
    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": metrics,
        },
        ckpt_dir / "training_state.pt"
    )

    console.print(f"[green]✓ Checkpoint saved[/] → {ckpt_dir}")


def train(data_path: str):
    """Main training function.

    Args:
        data_path: Path to the JSONL training dataset.
    """
    # ── Setup ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(Panel(
        f"[bold]Device:[/] {device}\n"
        f"[bold]VRAM:[/] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB "
        f"(if CUDA)\n"
        f"[bold]Data:[/] {data_path}\n"
        f"[bold]Epochs:[/] {CFG.training.num_epochs}\n"
        f"[bold]Grad accum:[/] {CFG.training.gradient_accumulation_steps} "
        f"(effective batch = {CFG.training.gradient_accumulation_steps})",
        title="🚀 Training Configuration",
        border_style="magenta",
    ))

    # ── Build model ──────────────────────────────────────────────────────
    from .model import build_model
    model, tokenizer = build_model()

    # ── Build data ───────────────────────────────────────────────────────
    train_loader, eval_loader = build_dataloaders(data_path, tokenizer)

    # ── Build optimizer + scheduler ──────────────────────────────────────
    total_steps = (
        len(train_loader)
        // CFG.training.gradient_accumulation_steps
        * CFG.training.num_epochs
    )
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, total_steps)

    # AMP gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.training.bf16 or CFG.training.fp16)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=CFG.training.logging_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = 0
    best_eval_loss = float("inf")

    for epoch in range(1, CFG.training.num_epochs + 1):
        console.rule(f"[bold blue]Epoch {epoch}/{CFG.training.num_epochs}[/]")

        lm_loss, aux_loss, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
            device=device,
        )

        # Evaluate
        eval_metrics = evaluate(model, eval_loader, device)
        writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
        writer.add_scalar("eval/perplexity", eval_metrics["eval_perplexity"], global_step)

        # Print epoch summary
        table = Table(title=f"Epoch {epoch} Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Train LM Loss", f"{lm_loss:.4f}")
        table.add_row("Train Aux Loss", f"{aux_loss:.4f}")
        table.add_row("Eval Loss", f"{eval_metrics['eval_loss']:.4f}")
        table.add_row("Eval Perplexity", f"{eval_metrics['eval_perplexity']:.2f}")
        table.add_row("Global Step", str(global_step))
        console.print(table)

        # Save checkpoint if eval improved
        if eval_metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["eval_loss"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, eval_metrics
            )

    writer.close()
    console.print(Panel(
        f"[bold green]Training complete![/]\n"
        f"Best eval loss: {best_eval_loss:.4f}\n"
        f"Checkpoints in: {CFG.training.output_dir}",
        title="✅ Done",
        border_style="green",
    ))


def main():
    """CLI entry point: python -m src.train"""
    import typer

    app = typer.Typer()

    @app.command()
    def run(
        data: str = typer.Option(
            CFG.distill.output_path,
            help="Path to JSONL training data"
        ),
        epochs: int = typer.Option(
            CFG.training.num_epochs,
            help="Number of training epochs"
        ),
    ):
        """Train the Mini-MoE-CoT student model."""
        CFG.training.num_epochs = epochs
        train(data)

    app()


if __name__ == "__main__":
    main()
