"""Root Mean Square Layer Normalization (RMSNorm) implementation."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)
    Formula: y = (x / sqrt(mean(x^2) + eps)) * weight

    Args:
        dim: Dimensionality of the input features to normalize (last dimension).
        eps: Small constant added to the variance denominator for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Computes RMS normalization in float32 for numerical stability."""
        # Upcast to float32 to prevent overflow/underflow in mixed precision
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return normed.to(input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMSNorm to input tensor x.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of the same shape and dtype as x.
        """
        return self._norm(x) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
