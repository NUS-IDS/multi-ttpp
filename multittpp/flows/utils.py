import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
from torch._prims_common import NumberType
from torch.cuda.amp import custom_bwd, custom_fwd  # automatic mixed precision package


class ClampPreserveGradients(torch.autograd.Function):
    """Clamp the tensor while preserving gradients in the clamped region.

    Implementation inspired by thread in PyTorch Discuss.
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    """

    @staticmethod
    @custom_fwd
    def forward(_, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(_, grad_output):
        return grad_output.clone(), None, None


@torch.jit.export
def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region."""
    return ClampPreserveGradients.apply(x, min, max)


@torch.jit.export
def inv_softplus(
    x: torch.Tensor, beta: Optional[NumberType] = None, threshold: NumberType = 20
) -> torch.Tensor:
    if beta is not None:
        scaled_input = beta * x
        y = x + torch.log(-torch.expm1(-scaled_input)) / beta
    else:
        scaled_input = x
        y = x + torch.log(-torch.expm1(-scaled_input))
    return torch.where(scaled_input > threshold, x, y)


@torch.jit.export
def new_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415

    From nano-GPT.
    https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class LayerNorm(nn.Module):
    """Layer norm with an optional bias. PyTorch does not support simply bias=False."""

    def __init__(self, n_embd, bias, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_embd, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
