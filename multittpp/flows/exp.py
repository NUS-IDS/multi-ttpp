import torch

from typing import Optional

from .base import Flow, FlowReturnType, InverseFlow
from .utils import clamp_preserve_gradients


class Exp(Flow):
    """Convert samples as y = exp(x).

    Args:
        n_marks (int): Total number of marks.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/exp.py
    """

    def __init__(self, *, n_marks: int, **kwargs) -> None:
        super().__init__(n_marks=n_marks, **kwargs)
        self.eps = 1e-10
        self.clamps = (self.eps, torch.inf)
        self.clamp_direction = "forward"

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        if y is None:
            y = x.exp()
            log_det_jac = x
        else:
            log_det_jac = None

        if xT is None:
            return y, log_det_jac

        yT = xT.exp()
        log_det_jacT = xT

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
            x = clamp_preserve_gradients(y, self.clamps[0], self.clamps[1]).log()
            inv_log_det_jac = -x
        else:
            inv_log_det_jac = None

        if yT is None:
            return x, inv_log_det_jac

        xT = clamp_preserve_gradients(yT, self.clamps[0], self.clamps[1]).log()
        inv_log_det_jacT = -xT

        return x, inv_log_det_jac, xT, inv_log_det_jacT


class Log(InverseFlow):
    """Convert samples as y = log(x).

    Args:
        n_marks (int): Total number of marks.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/exp.py
    """

    def __init__(self, *, n_marks: int, **kwargs) -> None:
        super().__init__(Exp, n_marks=n_marks, **kwargs)


class NegativeLog(Flow):
    """Convert samples as y = -log(x).

    Args:
        n_marks (int): Total number of marks.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/exp.py
    """

    def __init__(self, *, n_marks: int, **kwargs) -> None:
        super().__init__(n_marks=n_marks, **kwargs)
        self.eps = torch.finfo(torch.get_default_dtype()).eps
        self.clamps = (self.eps, 1.0 - self.eps)
        self.clamp_direction = "inverse"

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        """
        y = -log(x)
        log |dy/dx| = -x
        """
        if y is None:
            y = torch.where(
                k != self.n_marks,
                -clamp_preserve_gradients(x, self.clamps[0], self.clamps[1]).log(),
                x,
            )
            log_det_jac = y
        else:
            log_det_jac = None

        if xT is None:
            return y, log_det_jac

        yT = -clamp_preserve_gradients(xT, self.clamps[0], self.clamps[1]).log()
        log_det_jacT = yT

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
        """
        x = exp(-y)
        log |dx/dy| = x
        """
        if x is None:
            x = torch.where(k != self.n_marks, torch.exp(-y), y)
            inv_log_det_jac = -y
        else:
            inv_log_det_jac = None

        if yT is None:
            return x, inv_log_det_jac

        xT = torch.exp(-yT)
        inv_log_det_jacT = -yT

        return x, inv_log_det_jac, xT, inv_log_det_jacT


class ExpNegative(InverseFlow):
    """Convert samples as y = exp(-x).

    Args:
        n_marks (int): Total number of marks.


    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/exp.py
    """

    def __init__(self, *, n_marks: int, **kwargs) -> None:
        super().__init__(NegativeLog, n_marks=n_marks, **kwargs)
