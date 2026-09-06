"""SwiGLU Multi-Layer Perceptron (MLP) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sproutko.config import ModelConfig


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
    Formula: down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        config: Model configuration object.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies SwiGLU transformation to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # x: [B, T, hidden_size]
        # gate: [B, T, intermediate_size]
        # up:   [B, T, intermediate_size]
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        # down: [B, T, hidden_size]
        return self.down_proj(gate * up)
