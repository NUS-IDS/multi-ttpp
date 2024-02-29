import torch
import torch.nn.functional as F

from typing import Optional

from .base import Flow, FlowReturnType


class CumSum(Flow):
    """Computes mark-specific cumulative sum.

    Args:
        n_marks (int): Total number of marks.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/cumsum.py
    """

    def __init__(self, *, n_marks: int, **kwargs) -> None:
        super().__init__(n_marks=n_marks, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.LongTensor,
        xT: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        k_one_hot = F.one_hot(k, num_classes=self.n_marks + 1)[..., :-1]
        x_one_hot = x.unsqueeze(-1) * k_one_hot
        x_cumsum = x_one_hot.cumsum(-2)
        if y is None:
            y = (x_cumsum * k_one_hot).sum(dim=-1)
            log_det_jac = torch.zeros_like(y)
        else:
            log_det_jac = None

        if xT is None:
            return y, log_det_jac

        elif xT.dim() > 1 and xT.dim() < 4:
            B, _ = x.shape
            idxT = (k < self.n_marks).sum(dim=-1)
            yT = xT + F.pad(x_cumsum, (0, 0, 1, 0))[torch.arange(B), idxT]
            log_det_jacT = torch.zeros_like(xT)
            return y, log_det_jac, yT, log_det_jacT

        elif xT.dim() == 4:
            yT = xT + F.pad(x_one_hot, (0, 0, 1, -1)).cumsum(-2).unsqueeze(0)
            log_det_jacT = torch.zeros_like(yT)
            return y, log_det_jac, yT, log_det_jacT

        else:
            raise ValueError(
                f"yT shape is {yT.shape}, but such dimension is not allowed."
            )

    @torch.jit.export
    def inverse(
        self, y: torch.Tensor, k: torch.LongTensor, yT: Optional[torch.Tensor] = None
    ) -> FlowReturnType:
        y = F.pad(y, (1, 0))
        k = F.pad(k, (1, 0), value=self.n_marks)

        k_one_hot = F.one_hot(k, num_classes=self.n_marks + 1)[..., :-1]
        y_one_hot = y.unsqueeze(-1) * k_one_hot

        y_cummax, _ = y_one_hot.cummax(-2)

        x = (y_one_hot[:, 1:] - (y_cummax[:, :-1] * k_one_hot[:, 1:])).sum(-1)

        inv_log_det_jac = torch.zeros_like(x)

        if yT is None:
            return x, inv_log_det_jac

        elif yT.dim() > 1 and yT.dim() < 4:
            B, _ = y.shape
            idxT = (k < self.n_marks).sum(dim=-1)
            xT = yT - y_cummax[torch.arange(B), idxT]
            inv_log_det_jacT = torch.zeros_like(yT)
            return x, inv_log_det_jac, xT, inv_log_det_jacT

        elif yT.dim() == 4:
            xT = yT - y_cummax[None, :, :-1, :]
            inv_log_det_jacT = torch.zeros_like(xT)
            return x, inv_log_det_jac, xT, inv_log_det_jacT

        else:
            raise ValueError(
                f"yT shape is {yT.shape}, but such dimension is not allowed."
            )
