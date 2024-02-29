import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from ..config import Config
from .base import Flow, FlowReturnType


class BlockDiagonal(Flow):
    """Applies repeated mark-specific lower-triangular block diagonal matrices.

    Let H be the size of a lower-triangular block diagonal matrix A[k].
    This layer applies:
    [
        A[k], 0, 0, ...
        0, A[k], 0, ...
        0, 0, A[k], ...
        ., ., ., ...
    ]

    Args:
        block_size (int): Block size H.
        n_marks (int): Total number of marks.
        offset (int): Offset of A along the diagonal. Default: 0.


    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/block_diagonal.py
    """

    def __init__(
        self,
        *,
        n_marks: int,
        block_size: int = Config.block_size,
        offset: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(n_marks=n_marks, **kwargs)
        factory_kwargs = {k: v for k, v in kwargs.items() if k in ["device", "dtype"]}
        self.block_size = block_size
        self.n_params = (self.block_size**2 - self.block_size) // 2 + block_size
        self.params = nn.Parameter(
            torch.zeros((n_marks, self.n_params), **factory_kwargs)
        )
        mask_kwargs = factory_kwargs.copy()
        mask_kwargs.update({"dtype": torch.bool})
        mask = torch.zeros((block_size, block_size), **mask_kwargs)
        idx_kwargs = {k: v for k, v in kwargs.items() if k in ["device"]}
        idx_kwargs.update({"dtype": torch.long})
        idx = torch.tril_indices(block_size, block_size, **idx_kwargs)
        mask[tuple(idx)] = 1
        self.mask = nn.Parameter(mask, requires_grad=False)
        diag_idx = (torch.arange(block_size) + 1).cumsum(0) - 1
        self.diag_idx = nn.Parameter(diag_idx, requires_grad=False)
        self.offset = offset % self.block_size

    def _logdet(
        self,
        k: torch.Tensor,
        inputT: Optional[torch.Tensor],
        idxT: Optional[torch.Tensor],
        inverse: bool,
        compute_input: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the log|det(Jac(A(x)))|.

        Args:
            k (torch.Tensor): Marks, B x N/bs x bs.
            inverse (bool): Direction.

        Returns:
            torch.Tensor: log|det(Jac(A(x)))|.
        """
        B, N, _ = k.shape
        # the log determinant of the Jacobian corresponds to the diagonal
        # elements of the matrix
        params = torch.gather(self.params, 1, self.diag_idx.expand(self.n_marks, -1))

        if compute_input:
            # gather the parameter with mark corresponding to k
            logdet = torch.gather(
                params[None, :, None, :].expand(B, -1, N, -1),
                1,
                k.masked_fill(k == self.n_marks, 0).unsqueeze(1),
            )
            logdet = logdet.squeeze(1)
            logdet = logdet.masked_fill(k == self.n_marks, 0)
        else:
            logdet = None

        if inputT is None:
            logdetT = None
        else:
            if inputT.dim() > 1 and inputT.dim() < 4:
                # logdetT corresponds to all diagonal elements of the parameter at idxT
                logdetT = torch.gather(params, 1, idxT[1].expand(self.n_marks, -1))
                logdetT = logdetT.transpose(1, 0)
                if inputT.dim() == 3:
                    logdetT = logdetT.expand(inputT.shape[0], -1, -1)
            elif inputT.dim() == 5:
                logdetT = params.transpose(1, 0)[None, None, None, :, :].expand(
                    inputT.shape[0], B, N, -1, -1
                )
                logdetT = logdetT.masked_fill(
                    (k == self.n_marks)[None, :, :, :, None], 0
                )

        if not inverse:
            if compute_input:
                logdet = -logdet

            if inputT is not None:
                logdetT = -logdetT

        return logdet, logdetT

    def _add_padding(
        self, input: torch.Tensor, extra_pad: bool = False, value=0.0
    ) -> tuple[torch.Tensor, int]:
        """Applies padding to x such that x is a multiple of block_size.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Padded tensor.
            int: Size of applied padding to the end.
        """
        extra_offset = 1 if extra_pad else 0
        pad_end = self.block_size - (
            (input.shape[-1] + self.offset + extra_offset) % self.block_size
        )
        out = F.pad(input, (self.offset, extra_offset + pad_end), "constant", value)
        return out, pad_end + extra_offset

    def _matrices(self, inverse: bool = True) -> torch.Tensor:
        """Returns the matrix A.

        Args:
            inverse (bool): Whether to return the matrix A or its inverse A^-1.

        Returns:
            torch.Tensor: The matrix A or its inverse A^-1.
        """
        weights = self.params.clone()
        weights[:, self.diag_idx] = self.params[:, self.diag_idx].exp()

        A = torch.zeros(
            (self.n_marks, self.block_size, self.block_size), device=weights.device
        )
        A[:, self.mask] = weights
        if not inverse:
            A = torch.inverse(A)
        return A

    def _bmm(
        self,
        input: torch.Tensor,
        k: torch.Tensor,
        inputT: Optional[torch.Tensor],
        idxT: Optional[torch.Tensor],
        inverse: bool,
        out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Batched matrix multiplication.

        Args:
            x (torch.Tensor): Input, (B,N/bs,bs).
            k (torch.Tensor): Marks, (B, N/bs, bs).
            mat (torch.Tensor): Transformation matrix (bs,bs).

        Returns:
            torch.Tensor: Result of shape (B, N/bs, bs).
        """
        B, _, _ = input.shape

        invmat = self._matrices(inverse=True)

        if not inverse:
            mat = self._matrices(inverse=False)

        if out is None:
            if inverse:
                tmp = input.unsqueeze(1) @ invmat.transpose(-2, -1).unsqueeze(0)
                out = torch.gather(
                    tmp, 1, k.masked_fill(k == self.n_marks, 0).unsqueeze(1)
                ).squeeze(1)
                out = out.masked_fill(k == self.n_marks, 0)
            else:
                out = torch.empty_like(input)
                _k = k.masked_fill(k == self.n_marks, 0).unsqueeze(1)
                _input = torch.zeros(
                    (B, self.n_marks, input.shape[-2], input.shape[-1]),
                    dtype=input.dtype,
                    device=input.device,
                )
                _input = _input.scatter(1, _k, input.unsqueeze(1))
                for i in range(self.block_size):
                    tmp = _input @ mat.transpose(-2, -1).unsqueeze(0)[:, :, :, [i]]
                    out[:, :, [i]] = (
                        tmp.gather(1, _k[:, :, :, [i]]).squeeze(1)
                        * (k < self.n_marks)[:, :, [i]]
                    )
                    _input[:, :, :, [i]] = (
                        out.unsqueeze(1)
                        @ invmat.transpose(-2, -1).unsqueeze(0)[:, :, :, [i]]
                    )

                tmp = out.unsqueeze(1).expand(-1, self.n_marks, -1, -1)
                out = out.masked_fill(k == self.n_marks, 0)

        else:
            if inverse:
                raise NotImplementedError
            else:
                _input = out.unsqueeze(1) @ invmat.transpose(-2, -1).unsqueeze(0)
                tmp = out.unsqueeze(1).expand(-1, self.n_marks, -1, -1)

        if inputT is None:
            outT = None
        else:
            if inputT.dim() > 1 and inputT.dim() < 4:
                outT = tmp[torch.arange(B), :, idxT[0], idxT[1]]
                if inverse:
                    adj = inputT - input[torch.arange(B), idxT[0], idxT[1]].unsqueeze(1)
                    adj *= invmat.diagonal(dim1=-1, dim2=-2)[:, idxT[1]].transpose(1, 0)
                else:
                    adj = inputT - _input[torch.arange(B), :, idxT[0], idxT[1]]
                    adj *= mat.diagonal(dim1=-1, dim2=-2)[:, idxT[1]].transpose(1, 0)
                outT = outT + adj
            elif inputT.dim() == 5:
                outT = tmp
                if inverse:
                    adj = (
                        inputT.transpose(2, 4).transpose(-2, -1)
                        - input[None, :, None, :, :]
                    )
                    adj *= invmat.diagonal(dim1=-1, dim2=-2)[None, None, :, None, :]
                else:
                    adj = inputT.transpose(2, 4).transpose(-2, -1) - _input
                    adj *= mat.diagonal(dim1=-1, dim2=-2)[None, None, :, None, :]
                outT = outT + adj
                outT = outT.transpose(-2, -1).transpose(4, 2)

        return out, outT

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

        B, N = x.size()
        assert N >= self.block_size

        if xT is None:
            idxT = None
        elif xT.dim() > 1 and xT.dim() < 4:
            idxT = (k < self.n_marks).sum(dim=-1)
            idxT = idxT.masked_fill(idxT < 0, 0)
            idxT += self.offset
            idxT = torch.stack((idxT // self.block_size, idxT % self.block_size))
        elif xT.dim() == 4:
            idxT = None
            xT, _ = self._add_padding(xT.transpose(-2, -1))
            n_samples = xT.shape[0]
            xT = xT.transpose(-2, -1).view(
                n_samples, B, -1, self.block_size, self.n_marks
            )
        else:
            raise ValueError(
                f"xT shape is {xT.shape}, but such dimension is not allowed."
            )

        if (xT is not None) and (xT.dim() > 1) and (xT.dim() < 4):
            extra_pad = True
        else:
            extra_pad = False

        x, pad_end = self._add_padding(x, extra_pad=extra_pad)
        if not compute_y:
            y, _ = self._add_padding(y, extra_pad=extra_pad)
            y = y.view(B, -1, self.block_size)
        k, _ = self._add_padding(k, extra_pad=extra_pad, value=self.n_marks)
        x = x.view(B, -1, self.block_size)
        k = k.view(B, -1, self.block_size)

        y, yT = self._bmm(x, k, xT, idxT, inverse=False, out=y)

        logdet, logdetT = self._logdet(k, xT, idxT, False, compute_input=compute_y)

        y = y.reshape(B, -1)
        y = y[:, self.offset : -pad_end]
        if compute_y:
            logdet = logdet.reshape(B, -1)
            logdet = logdet[:, self.offset : -pad_end]

        if xT is not None and yT.dim() == 5:
            yT = yT.reshape(n_samples, B, -1, self.n_marks)
            yT = yT[:, :, self.offset : -pad_end, :]
            logdetT = logdetT.reshape(n_samples, B, -1, self.n_marks)
            logdetT = logdetT[:, :, self.offset : -pad_end, :]

        if xT is None:
            return y, logdet
        else:
            return y, logdet, yT, logdetT

    @torch.jit.export
    def inverse(
        self, y: torch.Tensor, k: torch.Tensor, yT: Optional[torch.Tensor] = None
    ) -> FlowReturnType:
        B, N = y.shape
        assert N >= self.block_size, f"N = {N} < self.block_size {self.block_size}"

        if yT is None:
            idxT = None
        elif yT.dim() > 1 and yT.dim() < 4:
            idxT = (k < self.n_marks).sum(dim=-1)
            idxT = idxT.masked_fill(idxT < 0, 0)
            idxT += self.offset
            idxT = torch.stack((idxT // self.block_size, idxT % self.block_size))
        elif yT.dim() == 4:
            idxT = None
            yT, _ = self._add_padding(yT.transpose(-2, -1))
            n_samples = yT.shape[0]
            yT = yT.transpose(-2, -1).view(
                n_samples, B, -1, self.block_size, self.n_marks
            )
        else:
            raise ValueError(
                f"yT shape is {yT.shape}, but such dimension is not allowed."
            )

        if (yT is not None) and (yT.dim() > 1) and (yT.dim() < 4):
            extra_pad = True
        else:
            extra_pad = False

        y, pad_end = self._add_padding(y, extra_pad=extra_pad)
        k, _ = self._add_padding(k, extra_pad=extra_pad, value=self.n_marks)
        y = y.view(B, -1, self.block_size)
        k = k.view(B, -1, self.block_size)

        x, xT = self._bmm(y, k, yT, idxT, inverse=True)

        logdet, logdetT = self._logdet(k, yT, idxT, True)

        x = x.reshape(B, -1)
        logdet = logdet.reshape(B, -1)
        x, logdet = x[:, self.offset : -pad_end], logdet[:, self.offset : -pad_end]
        if yT is not None and xT.dim() == 5:
            xT = xT.reshape(n_samples, B, -1, self.n_marks)
            logdetT = logdetT.reshape(n_samples, B, -1, self.n_marks)
            xT, logdetT = (
                xT[:, :, self.offset : -pad_end, :],
                logdetT[:, :, self.offset : -pad_end, :],
            )

        if yT is None:
            return x, logdet
        else:
            return x, logdet, xT, logdetT
