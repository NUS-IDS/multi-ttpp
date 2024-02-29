import torch
import torch.nn as nn

from typing import Optional, Type, TypeVar

FlowReturnType = TypeVar(
    "FlowReturnType",
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
)


class Flow(nn.Module):
    """Base class for flows.

    Args:
        n_marks (int): Total number of marks.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/base.py
    """

    n_marks: int

    def __init__(self, *, n_marks: int, **kwargs) -> None:
        super().__init__()
        self.n_marks = n_marks

    def update_n_max(self, n_events):
        pass

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        """
        Map compensated event time back to model event time.

        Args:
          x (torch.Tensor): Compensated event time, (B, N). Each element
              represents the duration in the compensated space since the last event
              with the same mark.
          k (torch.Tensor): Marks, (B, N). Each element represents the mark of
              the event.
          xT (torch.Tensor, optional): Compensated event time, (B, K) or (n_samples, B, K)
              or (n_samples, B, N, K). Compensated event time of last event in
              sequence for each mark. If xT has dimension (B, K) or (n_samples,
              B, K), the tensor holds the last event in the sequence given by
              x. Alternatively, if xT has dimension (n_samples, B, N, K), the
              tensor holds the last event in the cumulative given by x. That
              is, if x has 3 events, then xT holds the last events for
              subsequence starting with 0, 1, and 2 events.
        """
        raise NotImplementedError

    @torch.jit.export
    def inverse(
        self, y: torch.Tensor, k: torch.Tensor, yT: Optional[torch.Tensor] = None
    ) -> FlowReturnType:
        """
        Map model event time to compensated event time.

        Args:
          x (torch.Tensor): Model event time, (B, N). Each element
              represents the absolute model event time.
          k (torch.Tensor): Marks, (B, N). Each element represents the mark of
              the event.
          xT (torch.Tensor, optional): Model event time, (B, K) or (n_samples, B, K) or
              (n_samples, B, N, K). Model event time of last event in
              sequence for each mark. If yT has dimension (B, K) or (n_samples,
              B, K), the tensor holds the last event in the sequence given by
              y. Alternatively, if yT has dimension (n_samples, B, N, K), the
              tensor holds the last event in the cumulative given by y. That
              is, if y has 3 events, then yT holds the last events for
              subsequence starting with 0, 1, and 2 events.
        """
        raise NotImplementedError


class InverseFlow(Flow):
    """Base class for flows which are defined as the invert of another flow.

    Args:
        BaseFlow (multittpp.Flow): The base flow which will be inverted.
        n_marks (int): Total number of marks.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/base.py
    """

    def __init__(self, BaseFlow: Type[Flow], *, n_marks: int, **kwargs) -> None:
        nn.Module.__init__(self, **kwargs)
        self.base_flow = BaseFlow(n_marks=n_marks, **kwargs)
        self.n_marks = self.base_flow.n_marks
        if hasattr(self.base_flow, "domain"):
            self.codomain = self.base_flow.domain
        if hasattr(self.base_flow, "codomain"):
            self.domain = self.base_flow.codomain
        if hasattr(self.base_flow, "eps"):
            self.eps = self.base_flow.eps
        if hasattr(self.base_flow, "clamps"):
            self.clamps = self.base_flow.clamps
        if hasattr(self.base_flow, "clamp_direction"):
            if self.base_flow.clamp_direction == "forward":
                self.clamp_direction = "inverse"
            elif self.base_flow.clamp_direction == "inverse":
                self.clamp_direction = "forward"
            else:
                self.clamp_direciton = "both"

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> FlowReturnType:
        return self.base_flow._inverse(x, k, xT, y)

    @torch.jit.export
    def inverse(
        self, y: torch.Tensor, k: torch.Tensor, yT: Optional[torch.Tensor] = None
    ) -> FlowReturnType:
        return self.base_flow.forward(y, k, yT)
