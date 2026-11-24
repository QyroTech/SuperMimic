"""Explicit FP32 calibration outside compiled forwards and captured graphs."""

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor, nn

NetworkProfile = Literal["legacy", "running_silu"]
MLPPrecision = Literal["fp32", "bf16"]


class RunningNorm(nn.Module):
  mean: Tensor
  var: Tensor
  count: Tensor

  def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
    super().__init__()
    if dim < 1 or max_count <= 0 or eps <= 0:
      raise ValueError("Positive normalization dimension, count and epsilon required")
    self.max_count, self.eps = max_count, eps
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros(()))

  @torch.no_grad()
  def update(
    self,
    values: Tensor,
    reduce: Callable[[Tensor], Tensor] | None = None,
    valid: Tensor | None = None,
  ) -> None:
    if values.shape[-1] != self.mean.numel():
      raise ValueError("Normalizer feature dimension mismatch")
    values = values.detach().reshape(-1, self.mean.numel()).float()
    if valid is None:
      weights = values.new_ones(len(values))
    else:
      weights = valid.detach().reshape(-1).float()
      if len(weights) != len(values):
        raise ValueError("Normalizer mask dimension mismatch")
      weights = (weights > 0).float()
    values = torch.where(weights[:, None] > 0, values, 0)
    moments = torch.cat((weights.sum()[None], values.sum(0), values.square().sum(0)))
    if reduce is not None:
      moments = reduce(moments)
    n = moments[0]
    batch_mean = moments[1 : 1 + self.mean.numel()] / n.clamp_min(1)
    batch_var = (
      moments[1 + self.mean.numel() :] / n.clamp_min(1) - batch_mean.square()
    ).clamp_min(0)
    active = (self.count < self.max_count) & (n > 0)
    total = (self.count + n).clamp_min(1)
    delta = batch_mean - self.mean
    mean = self.mean + delta * n / total
    var = (
      self.var * self.count + batch_var * n + delta.square() * self.count * n / total
    ) / total
    # Keep buffer addresses stable for CUDA-graph replay and checkpoint keys.
    self.mean.copy_(torch.where(active, mean, self.mean))
    self.var.copy_(torch.where(active, var.clamp_min(0), self.var))
    self.count.copy_(torch.where(active, self.count + n, self.count))

  def std(self) -> Tensor:
    return (self.var.clamp_min(0) + self.eps**2).sqrt()

  def forward(self, values: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
    return (values.float() - self.mean) / self.std()

  def denormalize(self, values: Tensor) -> Tensor:
    return values.float() * self.std() + self.mean


def run_mlp(net: nn.Module, inputs: Tensor, precision: MLPPrecision) -> Tensor:
  """Autocast MLP work only; return FP32 for dynamics and losses."""
  if precision == "bf16" and inputs.is_cuda:
    with torch.autocast("cuda", dtype=torch.bfloat16):
      return net(inputs).float()
  return net(inputs)
