import torch
import torch.nn as nn

from typing import Optional

from .base import Flow, FlowReturnType


class Affine(Flow):
    """Element- and mark-wise affine transformation y[k] = a[k]*x[k] + b[k].

    Args:
        n_marks (int): Total number of marks.
        scale_init (float or torch.Tensor, optional): Initial value
            for the scale parameter a[k]. It can be a single shared parameter accross
            marks, or a tensor with a parameter for each mark. If None, no scale
            applied.
        shift_init (float or torch.Tensor, optional): Initial value
            for the shift parameter b[k]. It can be a single shared parameter
            accross marks, or a tensor with a parameter for each mark. If None, no
            shift applied.
        trainable (bool): Make the transformation parameters a and b learnable. Default: True.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/affine.py
    """

    def __init__(
        self,
        *,
        n_marks: int,
        scale_init: Optional[float | torch.Tensor] = None,
        shift_init: Optional[float | torch.Tensor] = None,
        trainable: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(n_marks=n_marks, **kwargs)
        factory_kwargs = {k: v for k, v in kwargs.items() if k in ["device", "dtype"]}

        if scale_init is not None:
            if isinstance(scale_init, torch.Tensor):
                assert scale_init.shape == torch.Size(
                    (n_marks,)
                ), f"scale_init size {scale_init.shape} does not match n_marks {n_marks}"
                self.log_scale = nn.Parameter(scale_init.log(), requires_grad=trainable)
            else:
                if scale_init == 1 and not trainable:
                    self.register_parameter("log_scale", None)
                else:
                    log_scale_init = torch.tensor([scale_init], **factory_kwargs).log()
                    self.log_scale = nn.Parameter(
                        log_scale_init, requires_grad=trainable
                    )
        else:
            self.register_parameter("log_scale", None)

        if shift_init is not None:
            if isinstance(shift_init, torch.Tensor):
                assert shift_init.shape == torch.Size(
                    (n_marks,)
                ), f"shift_init size {shift_init.shape} does not match n_marks {n_marks}"
                self.shift = nn.Parameter(shift_init, requires_grad=trainable)
            else:
                if shift_init == 0 and not trainable:
                    self.register_parameter("shift", None)
                else:
                    shift_init = torch.tensor(
                        [shift_init], dtype=torch.get_default_dtype()
                    )
                    self.shift = nn.Parameter(shift_init, requires_grad=trainable)
        else:
            self.register_parameter("shift", None)

    def _scale(
        self, input: torch.Tensor, mark: Optional[int] = None, inverse: bool = True
    ) -> torch.Tensor:
        if self.log_scale is None:
            return input

        if self.log_scale.shape[0] == 1 or mark is None:
            if inverse:
                out = input * torch.exp(-self.log_scale)
            else:
                out = input * torch.exp(self.log_scale)
        else:
            if inverse:
                out = input * torch.exp(-self.log_scale[mark])
            else:
                out = input * torch.exp(self.log_scale[mark])

        return out

    def _shift(
        self, input: torch.Tensor, mark: Optional[int] = None, inverse: bool = True
    ) -> torch.Tensor:
        if self.shift is None:
            return input

        if self.shift.shape[0] == 1 or mark is None:
            if inverse:
                out = input - self.shift
            else:
                out = input + self.shift
        else:
            if inverse:
                out = input - self.shift[mark]
            else:
                out = input + self.shift[mark]

        return out

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        if y is None:
            compute_y = True
        else:
            compute_y = False
            log_det_jac = None

        if (self.shift is not None and self.shift.shape[0] > 1) or (
            self.log_scale is not None and self.log_scale.shape[0] > 1
        ):
            if compute_y:
                y = torch.empty_like(x)
                y[k == self.n_marks] = 0
                if self.log_scale is not None and self.log_scale.shape[0] > 1:
                    log_det_jac = torch.empty_like(x)
                    log_det_jac[k == self.n_marks] = 0

            if xT is not None:
                yT = torch.empty_like(xT)
                if self.log_scale is not None and self.log_scale.shape[0] > 1:
                    log_det_jacT = torch.empty_like(xT)

            for mark in range(self.n_marks):
                if compute_y:
                    y_mark = x.clone()
                    y_mark = self._scale(y_mark, mark, inverse=False)
                    y_mark = self._shift(y_mark, mark, inverse=False)
                    mask = (k == mark).squeeze(-1)
                    y[mask] = y_mark[mask]
                    if self.log_scale is not None and self.log_scale.shape[0] > 1:
                        log_det_jac[mask] = self.log_scale[mark]

                if xT is not None:
                    yT_mark = xT[..., mark].clone()
                    yT_mark = self._scale(yT_mark, mark, inverse=False)
                    yT_mark = self._shift(yT_mark, mark, inverse=False)
                    yT[..., mark] = yT_mark
                    if self.log_scale is not None and self.log_scale.shape[0] > 1:
                        log_det_jacT[..., mark] = self.log_scale[mark]
        else:
            if compute_y:
                y = x.clone()
                y = self._scale(y, inverse=False)
                y = self._shift(y, inverse=False)

            if xT is not None:
                yT = xT.clone()
                yT = self._scale(yT, inverse=False)
                yT = self._shift(yT, inverse=False)

        if compute_y:
            if self.log_scale is None:
                log_det_jac = torch.zeros_like(y)
            elif self.log_scale.shape[0] == 1:
                log_det_jac = self.log_scale.expand(y.shape)

        if xT is not None:
            if self.log_scale is None:
                log_det_jacT = torch.zeros_like(xT)
            elif self.log_scale.shape[0] == 1:
                log_det_jacT = self.log_scale.expand(xT.shape)

        if xT is None:
            return y, log_det_jac
        else:
            return y, log_det_jac, yT, log_det_jacT

    @torch.jit.export
    def inverse(
        self,
        y: torch.Tensor,
        k: torch.Tensor,
        yT: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        if (self.shift is not None and self.shift.shape[0] > 1) or (
            self.log_scale is not None and self.log_scale.shape[0] > 1
        ):
            x = torch.empty_like(y)
            x[k == self.n_marks] = 0
            if self.log_scale is not None and self.log_scale.shape[0] > 1:
                inv_log_det_jac = torch.empty_like(y)
                inv_log_det_jac[k == self.n_marks] = 0

            if yT is not None:
                xT = torch.empty_like(yT)
                if self.log_scale is not None and self.log_scale.shape[0] > 1:
                    inv_log_det_jacT = torch.empty_like(yT)

            for mark in range(self.n_marks):
                x_mark = y.clone()
                x_mark = self._shift(x_mark, mark)
                x_mark = self._scale(x_mark, mark)
                mask = (k == mark).squeeze(-1)
                x[mask] = x_mark[mask]
                if self.log_scale is not None and self.log_scale.shape[0] > 1:
                    inv_log_det_jac[mask] = -self.log_scale[mark]

                if yT is not None:
                    xT_mark = yT[..., mark].clone()
                    xT_mark = self._shift(xT_mark, mark)
                    xT_mark = self._scale(xT_mark, mark)
                    xT[..., mark] = xT_mark
                    if self.log_scale is not None and self.log_scale.shape[0] > 1:
                        inv_log_det_jacT[..., mark] = -self.log_scale[mark]
        else:
            x = y.clone()
            x = self._shift(x)
            x = self._scale(x)

            if yT is not None:
                xT = yT.clone()
                xT = self._shift(xT)
                xT = self._scale(xT)

        if self.log_scale is None:
            inv_log_det_jac = torch.zeros_like(y)
        elif self.log_scale.shape[0] == 1:
            inv_log_det_jac = -self.log_scale.expand(y.shape)

        if yT is not None:
            if self.log_scale is None:
                inv_log_det_jacT = torch.zeros_like(yT)
            elif self.log_scale.shape[0] == 1:
                inv_log_det_jacT = -self.log_scale.expand(yT.shape)

        if yT is None:
            return x, inv_log_det_jac
        else:
            return x, inv_log_det_jac, xT, inv_log_det_jacT
