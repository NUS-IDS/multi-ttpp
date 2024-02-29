import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from ..config import Config
from .base import FlowReturnType
from .spline import Spline
from .utils import (
    new_gelu,
    LayerNorm,
)


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer.

    Implementation inspired by nano-GPT.
    https://github.com/karpathy/nanoGPT/blob/master/model.py
    """

    def __init__(
        self,
        *,
        n_embd: int = Config.n_embd,
        n_heads: int = Config.n_heads,
        dropout: float = Config.dropout,
        bias: bool = True,
    ):
        assert n_embd % n_heads == 0
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            n_embd,
            3 * n_embd,
            bias=bias,
        )
        self.c_proj = nn.Linear(
            n_embd,
            n_embd,
            bias=bias,
        )
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = dropout
        # self.alphas = nn.Parameter(torch.randn((n_heads,), requires_grad = True))

    # def _time_decay(self, lag_matrix: torch.Tensor):
    #     return (-self.alphas[None, :, None, None].square() * lag_matrix[:, None, :, :]).exp()

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        (
            B,
            N,
            E,
        ) = (
            input.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(input).split(self.n_embd, dim=2)
        k = k.view(B, N, self.n_heads, E // self.n_heads).transpose(
            1, 2
        )  # (B, nh, N, hs)
        q = q.view(B, N, self.n_heads, E // self.n_heads).transpose(
            1, 2
        )  # (B, nh, N, hs)
        v = v.view(B, N, self.n_heads, E // self.n_heads).transpose(
            1, 2
        )  # (B, nh, N, hs)

        # causal self-attention: (B, nh, N, hs) x (B, nh, hs, N) -> (B, nh, N, N)
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        # equivalent to manual implementation of the following
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # mask = (
        #     torch.ones(N, N, dtype=torch.bool, device=q.device)
        #     .tril(diagonal=0)
        # )
        # att = att.masked_fill(~mask, float("-inf"))
        # att = att * self._time_decay(input)
        # att = F.softmax(att, dim=-1)
        # att = self.attn_dropout(att)
        # y = att @ v # (B, nh, N, N) x (B, nh, N, hs) -> (B, nh, N, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, N, E)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerMLP(nn.Module):
    """Multi-layer perceptron.

    Implementation inspired by nano-GPT.
    https://github.com/karpathy/nanoGPT/blob/master/model.py
    """

    def __init__(
        self,
        *,
        n_embd: int = Config.n_embd,
        bias: bool = True,
        dropout: float = Config.dropout,
    ) -> None:
        super().__init__()
        self.c_fc = nn.Linear(
            n_embd,
            4 * n_embd,
            bias=bias,
        )
        self.c_proj = nn.Linear(
            4 * n_embd,
            n_embd,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        input = self.c_fc(input)
        input = new_gelu(input)
        input = self.c_proj(input)
        return self.dropout(input)


class TransformerBlock(nn.Module):
    """Transformer block.

    Implementation inspired by nano-GPT.
    https://github.com/karpathy/nanoGPT/blob/master/model.py
    """

    def __init__(
        self,
        *,
        n_embd: int = Config.n_embd,
        n_heads: int = Config.n_heads,
        dropout: float = Config.dropout,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.layer_norm_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
        )
        self.layer_norm_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = TransformerMLP(
            n_embd=n_embd,
            bias=bias,
            dropout=dropout,
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        input = input + self.attn(self.layer_norm_1(input))
        input = input + self.mlp(self.layer_norm_2(input))
        return input


class TimeEmbedding(nn.Module):
    """Time embedding.

    Implementation inspired by GNTPP.
    https://github.com/BIRD-TAO/GNTPP/blob/main/models/embedding/time.py
    """

    def __init__(self, *, n_embd: int):
        super().__init__()
        assert n_embd % 2 == 0
        self.linear = nn.Linear(1, n_embd // 2, bias=False)

    def forward(self, input: torch.Tensor):
        phi = self.linear(input.unsqueeze(-1))
        pe_sin = torch.sin(phi)
        pe_cos = torch.cos(phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe


class PositionEmbedding(nn.Module):
    """Position embedding.

    Implementation inspired by GNTPP.
    https://github.com/BIRD-TAO/GNTPP/blob/main/models/embeddings/position.py
    """

    def __init__(self, *, n_embd: int, n_events: int):
        super().__init__()
        self.n_embd = n_embd
        self.register_buffer(
            "embedding", self._build_embedding(n_events, n_embd), persistent=False
        )
        self.linear1 = nn.Linear(n_embd * 2, n_embd)
        self.linear2 = nn.Linear(n_embd, n_embd)

    def forward(self, input: torch.Tensor):
        out = self.embedding[input]
        out = self.linear1(out)
        out = F.silu(out)
        out = self.linear2(out)
        out = F.silu(out)
        return out

    def _build_embedding(self, n_events: int, n_embd: int, device: Optional = None):
        pos = torch.arange(n_events + 1, device=device).unsqueeze(1)  # [N, 1]
        embd = torch.arange(1, n_embd + 1, device=device).unsqueeze(0)  # [1, E]
        table = pos * 10.00 ** (embd * 4.0 / embd)  # [N, E]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

    def update_n_max(self, n_events):
        with torch.no_grad():
            self.embedding = self._build_embedding(
                n_events + 1, self.n_embd, device=self.embedding.device
            )


class AutoregressiveTransformer(nn.Module):
    """Autoregressive transformer model.

    Implementation inspired by nano-GPT.
    https://github.com/karpathy/nanoGPT/blob/master/model.py

    Implementation inspired by GNTPP.
    https://github.com/BIRD-TAO/GNTPP/blob/main/models/hist_encoders/attention.py
    """

    def __init__(
        self,
        *,
        n_marks: int,
        n_events: int,
        n_out: int = 1,
        n_embd: int = Config.n_embd,
        n_heads: int = Config.n_heads,
        n_blocks: int = Config.n_blocks,
        dropout: float = Config.dropout,
        bias=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_marks = n_marks
        self.transformer = nn.ModuleDict(
            dict(
                wte=TimeEmbedding(n_embd=n_embd),
                wpe=PositionEmbedding(n_embd=n_embd, n_events=n_events),
                wme=nn.Embedding(n_marks + 1, n_embd, padding_idx=n_marks),
                drop=nn.Dropout(dropout),
                blocks=nn.ModuleList(
                    [
                        TransformerBlock(
                            n_embd=n_embd,
                            n_heads=n_heads,
                            dropout=dropout,
                            bias=bias,
                        )
                        for _ in range(n_blocks)
                    ]
                ),
                layer_norm=LayerNorm(n_embd, bias=bias),
            )
        )
        self.linear = nn.Linear(n_embd, n_out)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))

    def _init_weights(self, layer: nn.Module):
        if isinstance(layer, (nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.Embedding)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def update_n_max(self, n_events):
        self.transformer["wpe"].update_n_max(n_events)

    def forward(
        self,
        input: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, N = input.size()
        time_emb = self.transformer["wte"](input)
        pos_emb = self.transformer["wpe"](torch.arange(N, device=input.device))
        mark_emb = self.transformer["wme"](k)
        out = self.transformer["drop"](pos_emb + time_emb + mark_emb)
        for block in self.transformer.blocks:
            out = block(out)
        out = self.transformer["layer_norm"](out)
        out = self.linear(out)
        return out.squeeze(-1)


class SplineTransformer(Spline):
    """Transformer produces the parameter for a mark-specific spline flow."""

    def __init__(
        self,
        *,
        n_marks: int,
        n_events: int,
        n_embd: int = Config.n_embd,
        n_heads: int = Config.n_heads,
        n_blocks: int = Config.n_blocks,
        dropout: float = Config.dropout,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(n_marks=n_marks, **kwargs)
        del self.params
        self.autoregressive_transformer = AutoregressiveTransformer(
            n_marks=n_marks,
            n_events=n_events,
            n_out=self.n_params * (n_marks + 1),
            n_embd=n_embd,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
            bias=bias,
            **kwargs,
        )
        self.linear = nn.Linear(n_marks, self.n_params * n_marks)
        self.eps = torch.finfo(torch.get_default_dtype()).eps

    def update_n_max(self, n_events):
        self.autoregressive_transformer.update_n_max(n_events)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        B, N = x.shape

        # run the transformer iteratively, one point at a time
        if y is None:
            y = torch.zeros((B, N), device=x.device)
            log_det_jac = torch.zeros(B, N, device=x.device)
            for i in range(N):
                _y = F.pad(y, (1, 0, 0, 0))
                _k = F.pad(k, (1, 0, 0, 0), value=self.n_marks)
                _k[:, (i + 1) :] = self.n_marks
                params = self.autoregressive_transformer(_y, _k)
                params = params.view(B, N + 1, (self.n_marks + 1), -1)

                k_one_hot = F.one_hot(k, num_classes=self.n_marks + 1)
                params_x = (k_one_hot.unsqueeze(-1) * params[:, :-1]).sum(-2)

                y[:, i], log_det_jac[:, i] = self._unconstrained_spline(
                    x[:, i], k[:, i], params_x[:, i]
                )
        else:
            log_det_jac = None
            _y = F.pad(y, (1, 0, 0, 0))
            _k = F.pad(k, (1, 0, 0, 0), value=self.n_marks)
            params = self.autoregressive_transformer(_y, _k)
            params = params.view(B, N + 1, (self.n_marks + 1), -1)

        if xT is None:
            return y, log_det_jac

        if xT.dim() == 2:
            kT = torch.arange(self.n_marks, device=k.device).expand(B, -1)
            idxT = (_k < self.n_marks).sum(dim=-1)
            idxT = idxT.masked_fill(idxT < 0, 0)
            params_xT = params[torch.arange(B), idxT, : self.n_marks]
        elif xT.dim() == 3:
            n_samples, _, _ = xT.shape
            kT = torch.arange(self.n_marks, device=k.device).expand(B, -1)
            idxT = (_k < self.n_marks).sum(dim=-1)
            idxT = idxT.masked_fill(idxT < 0, 0)
            params_xT = params[torch.arange(B), idxT, : self.n_marks]
            params_xT = params_xT.expand(xT.shape[0], -1, -1, -1)
        elif xT.dim() == 4:
            n_samples, _, _, _ = xT.shape
            kT = torch.arange(self.n_marks, device=k.device).expand(
                n_samples, B, N, self.n_marks
            )
            params_xT = params[:, :-1, : self.n_marks].expand(n_samples, -1, -1, -1, -1)
        else:
            raise ValueError(
                f"xT shape is {xT.shape}, but such dimension is not allowed."
            )

        yT, log_det_jacT = self._unconstrained_spline(xT, kT, params_xT)

        return y, log_det_jac, yT, log_det_jacT

    @torch.jit.export
    def inverse(
        self, y: torch.Tensor, k: torch.Tensor, yT: Optional[torch.Tensor] = None
    ) -> FlowReturnType:
        B, N = y.shape
        _y = F.pad(y, (1, 0, 0, 0))
        _k = F.pad(k, (1, 0, 0, 0), value=self.n_marks)
        params = self.autoregressive_transformer(_y, _k)
        params = params.view(B, N + 1, (self.n_marks + 1), -1)

        k_one_hot = F.one_hot(k, num_classes=self.n_marks + 1)
        params_y = (k_one_hot.unsqueeze(-1) * params[:, :-1]).sum(-2)

        x, inv_log_det_jac = self._unconstrained_spline(
            y,
            k,
            params_y,
            inverse=True,
        )

        if yT is None:
            return x, inv_log_det_jac

        if yT.dim() == 2:
            kT = torch.arange(self.n_marks, device=k.device).expand(B, -1)
            idxT = (k < self.n_marks).sum(dim=-1)
            params_yT = params[torch.arange(B), idxT, : self.n_marks]
        elif yT.dim() == 3:
            n_samples, _, _ = yT.shape
            kT = torch.arange(self.n_marks, device=k.device).expand(n_samples, B, -1)
            idxT = (k < self.n_marks).sum(dim=-1)
            params_yT = params[torch.arange(B), idxT, : self.n_marks]
            params_yT = params_yT.expand(n_samples, -1, -1, -1)
        elif yT.dim() == 4:
            n_samples, _, _, _ = yT.shape
            kT = torch.arange(self.n_marks, device=k.device).expand(
                n_samples, B, N, self.n_marks
            )
            params_yT = params[:, :-1, : self.n_marks].expand(n_samples, -1, -1, -1, -1)
        else:
            raise ValueError(
                f"yT shape is {yT.shape}, but such dimension is not allowed."
            )

        xT, inv_log_det_jacT = self._unconstrained_spline(
            yT, kT, params_yT, inverse=True
        )

        return x, inv_log_det_jac, xT, inv_log_det_jacT
