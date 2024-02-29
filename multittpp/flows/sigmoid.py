import torch
import torch.nn.functional as F

from typing import Optional

from .base import Flow, FlowReturnType, InverseFlow
from .utils import clamp_preserve_gradients


class Sigmoid(Flow):
    """Computes 1/(1+e^(-x)).

    Args:
        n_marks (int): Total number of marks.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/sigmoid.py
    """

    def __init__(self, *, n_marks: int, **kwargs) -> None:
        super().__init__(n_marks=n_marks, **kwargs)
        self.eps = 1e-6
        # upper clamp set after manually experimenting
        self.clamps = (self.eps, 0.999)
        self.clamp_direction = "forward"

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        if y is None:
            y = torch.sigmoid(x)
            log_det_jac = -F.softplus(-x) - F.softplus(x)
        else:
            log_det_jac = None

        if xT is None:
            return y, log_det_jac

        yT = torch.sigmoid(xT)
        log_det_jacT = -F.softplus(-xT) - F.softplus(xT)

        return y, log_det_jac, yT, log_det_jacT

    @torch.jit.export
    def inverse(
        self, y: torch.Tensor, k: torch.Tensor, yT: Optional[torch.Tensor] = None
    ) -> FlowReturnType:
        return self._inverse(y, k, yT)

    @torch.jit.export
    def _inverse(
        self,
        y: torch.Tensor,
        k: torch.Tensor,
        yT: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        if x is None:
            y = clamp_preserve_gradients(y, self.clamps[0], self.clamps[1])
            x = torch.log(y) - torch.log1p(-y)
            inv_log_det_jac = -torch.log(y) - torch.log1p(-y)
        else:
            inv_log_det_jac = None

        if yT is None:
            return x, inv_log_det_jac

        yT = clamp_preserve_gradients(yT, self.clamps[0], self.clamps[1])
        xT = torch.log(yT) - torch.log1p(-yT)
        inv_log_det_jacT = -torch.log(yT) - torch.log1p(-yT)

        return x, inv_log_det_jac, xT, inv_log_det_jacT


class Logit(InverseFlow):
    """Computes ln(x/(1-x)).

    Args:
        n_marks (int): Total number of marks.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/sigmoid.py
    """

    def __init__(self, *, n_marks: int, **kwargs) -> None:
        return super().__init__(Sigmoid, n_marks=n_marks, **kwargs)
