"""
moe_layer.py — Mixture-of-Experts layer implementation.

This module implements the core MoE mechanism:
  1. A learned Router that assigns tokens to experts
  2. A pool of Expert FFN networks
  3. Weighted combination of expert outputs
  4. Auxiliary load-balancing loss to prevent expert collapse

Architecture (per token):
  hidden_state → Router → top-k expert indices + weights
                        → Expert_i(hidden_state) for each selected i
                        → weighted sum of expert outputs

Why MoE instead of one big FFN?
  - Same parameter count, but only a fraction (top_k/num_experts) are
    active per token → faster inference.
  - Different experts specialize in different token patterns over training.
  - Scalable: add more experts without proportionally more compute.

Reference: "Outrageously Large Neural Networks: The Sparsely-Gated MoE"
           (Shazeer et al., 2017) and DeepSeek-V3 technical report.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """A single FFN expert.

    Identical in structure to a standard transformer FFN block:
      Linear → SiLU → Linear (with a gating variant: SwiGLU)

    SwiGLU is used in Qwen/LLaMA models and performs better than
    plain ReLU for language tasks.

    Args:
        hidden_size: Input/output dimension (must match base model).
        intermediate_size: Inner dimension of the FFN.
        dropout: Dropout probability during training.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # SwiGLU: gate_proj and up_proj are applied in parallel,
        # then element-wise multiplied before down_proj.
        # This gating mechanism improves gradient flow.
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
               OR (num_selected_tokens, hidden_size) after token scattering.

        Returns:
            Output tensor of same shape as input.
        """
        # SwiGLU: gate * SiLU(gate_proj) element-wise with up_proj output
        gate = F.silu(self.gate_proj(x))   # shape: (..., intermediate_size)
        up = self.up_proj(x)               # shape: (..., intermediate_size)
        return self.dropout(self.down_proj(gate * up))


class Router(nn.Module):
    """Learned token router for MoE.

    The router is a simple linear projection from hidden_size → num_experts,
    followed by a softmax. The top-k experts with the highest scores are
    selected for each token.

    Key design choices:
    - No bias in the projection (prevents routing bias toward specific experts).
    - Softmax scores are used as weights to combine expert outputs (not just
      for selection) — this makes the operation differentiable end-to-end.

    Args:
        hidden_size: Input dimension.
        num_experts: Total number of available experts.
        top_k: Number of experts activated per token.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights and selected expert indices.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            routing_weights: (batch_size, seq_len, top_k) — softmax weights
                             for selected experts. Sum to 1 across top_k dim.
            selected_experts: (batch_size, seq_len, top_k) — indices of the
                              top-k experts chosen for each token.
            router_logits: (batch_size, seq_len, num_experts) — raw logits
                           before top-k selection. Needed for aux loss.
        """
        # Raw expert scores for each token: (B, T, num_experts)
        router_logits = self.gate(hidden_states)

        # Select top-k experts per token
        # topk_weights: (B, T, top_k)   — raw scores of selected experts
        # topk_indices: (B, T, top_k)   — which experts were selected
        topk_weights, topk_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        # Normalize the selected weights so they sum to 1.
        # This lets us take a weighted average of expert outputs.
        routing_weights = F.softmax(topk_weights, dim=-1)

        return routing_weights, topk_indices, router_logits


def compute_aux_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    aux_loss_coef: float,
) -> torch.Tensor:
    """Compute load-balancing auxiliary loss.

    Without this, the router tends to collapse: it finds a few "safe"
    experts and ignores the rest, wasting parameters.

    This loss penalizes uneven expert utilization. It is the product of:
    - Expert fraction f_i: fraction of tokens routed to expert i
    - Expert probability P_i: average routing probability for expert i

    We want both to be uniform (= 1/num_experts each).

    Reference: Switch Transformer (Fedus et al., 2021), equation 4.

    Args:
        router_logits: (batch*seq_len, num_experts) raw router scores.
        num_experts: Total number of experts.
        top_k: Number of active experts per token.
        aux_loss_coef: Weight of the auxiliary loss term.

    Returns:
        Scalar auxiliary loss tensor.
    """
    # Flatten batch and sequence dimensions
    # router_logits: (N, num_experts) where N = batch * seq_len
    if router_logits.dim() == 3:
        router_logits = router_logits.view(-1, num_experts)

    num_tokens = router_logits.shape[0]

    # Routing probabilities (after softmax): (N, num_experts)
    routing_probs = F.softmax(router_logits, dim=-1)

    # P_i: mean probability assigned to each expert across all tokens
    # Shape: (num_experts,)
    expert_probs = routing_probs.mean(dim=0)

    # f_i: fraction of tokens for which expert i is in the top-k
    # We one-hot encode top-k selections and count.
    _, top_indices = torch.topk(router_logits, top_k, dim=-1)
    # One-hot: (N, num_experts)
    expert_mask = torch.zeros(
        num_tokens, num_experts, device=router_logits.device
    )
    expert_mask.scatter_(1, top_indices, 1.0)
    # f_i: fraction of tokens dispatched to each expert
    expert_fraction = expert_mask.mean(dim=0)

    # Aux loss = num_experts * sum(f_i * P_i)
    # When routing is uniform, each f_i = P_i = 1/num_experts,
    # so loss = num_experts * num_experts * (1/num_experts)^2 = 1.
    # Any deviation increases the loss.
    aux_loss = aux_loss_coef * num_experts * (expert_fraction * expert_probs).sum()
    return aux_loss


class MoELayer(nn.Module):
    """Full Mixture-of-Experts layer.

    Replaces or augments a single FFN layer in a transformer.
    Each token is routed to top_k out of num_experts expert FFNs,
    and outputs are combined with learned routing weights.

    Token dispatch strategy: "token-choice" (each token picks its own
    experts), as opposed to "expert-choice" (each expert picks tokens).
    Token-choice is simpler and matches DeepSeek/Mixtral behavior.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: Expert FFN inner dimension.
        num_experts: Total number of experts.
        top_k: Experts activated per token.
        aux_loss_coef: Load-balancing loss weight.
        expert_dropout: Dropout inside each expert.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        expert_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        # Router: assigns each token to top-k experts
        self.router = Router(hidden_size, num_experts, top_k)

        # Expert pool: num_experts independent FFN networks
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size, expert_dropout)
            for _ in range(num_experts)
        ])

        # Layer norm before the MoE (pre-norm architecture, like Qwen)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MoE layer.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            output: (batch_size, seq_len, hidden_size) — combined expert output
            aux_loss: Scalar load-balancing loss to add to training loss.

        Implementation note:
            We iterate over experts explicitly (not batched dispatch).
            This is simpler to understand and correct for small num_experts,
            but not optimal for production. For scale, use `torch.einsum`
            or a proper scatter/gather dispatch kernel.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Pre-norm (add residual after)
        normed = self.norm(hidden_states)

        # Get routing decisions
        # routing_weights: (B, T, top_k)
        # selected_experts: (B, T, top_k) — integer expert indices
        # router_logits: (B, T, num_experts) — for aux loss
        routing_weights, selected_experts, router_logits = self.router(normed)

        # Accumulator for the weighted sum of expert outputs
        output = torch.zeros_like(normed)

        # Iterate over each expert position in the top-k selection
        # (not over num_experts — we only compute what's needed)
        for k in range(self.top_k):
            # expert_idx_for_k: (B, T) — which expert is in position k
            expert_idx_for_k = selected_experts[:, :, k]       # (B, T)
            # weight_for_k: (B, T) — the softmax weight for this expert
            weight_for_k = routing_weights[:, :, k].unsqueeze(-1)  # (B, T, 1)

            # For each expert, gather the tokens it handles and run them
            for expert_id in range(self.num_experts):
                # Mask: which (batch, token) positions chose this expert
                # at position k in their top-k selection
                mask = (expert_idx_for_k == expert_id)   # (B, T) bool

                if not mask.any():
                    continue  # This expert not selected at position k

                # Extract tokens for this expert
                # selected_tokens: (num_selected, hidden_size)
                selected_tokens = normed[mask]

                # Run expert FFN
                expert_output = self.experts[expert_id](selected_tokens)

                # Weight and accumulate (scatter back to full tensor)
                output[mask] += weight_for_k.expand_as(normed)[mask] * expert_output

        # Residual connection
        output = hidden_states + output

        # Compute auxiliary load-balancing loss
        aux_loss = compute_aux_loss(
            router_logits, self.num_experts, self.top_k, self.aux_loss_coef
        )

        return output, aux_loss
