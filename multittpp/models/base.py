from warnings import warn
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from multittpp.data import Batch

from ..flows.base import Flow, FlowReturnType
from ..config import Config


class StandardExponential(nn.Module):
    """Standard exponential distribution with unit rate.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/models.py
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_survival(self, x: torch.Tensor) -> torch.Tensor:
        result = -x
        result[x < 0] = 0.0
        return result

    def quantile(self, p: float) -> torch.Tensor:
        return -torch.tensor(1 - p).log()

    def pdf(self, q: torch.Tensor) -> torch.Tensor:
        return (-q).exp()

    def cdf(self, q: torch.Tensor) -> torch.Tensor:
        out = 1 - (-q).exp()
        return out.masked_fill(out < 0, 0)

    def sample(
        self, sample_shape: tuple[int], *, device: Optional[str | torch.device]
    ) -> torch.Tensor:
        return torch.empty(sample_shape, device=device).exponential_()

    def rsample(
        self,
        sample_shape: tuple[int],
        *,
        cond: Optional[torch.Tensor] = None,
        device: Optional[str | torch.device],
    ) -> torch.Tensor:
        """Sample via re-parameterization trick."""
        out = self.sample(sample_shape, device=device)
        if cond is not None:
            # memoryless property implies that the conditional sample is simpy
            # the conditional plus sample
            out += cond
        return out


class TransformedExponential(nn.Module):
    """Base class for all the models considered in the paper (except Hawkes process).

    We parametrize TPP densities by specifying the sequence of transformations that
    convert an arbitrary TPP into a homogeneous Poisson process with unit rate.

    There are two small differences compared to the notation used in the paper:
    1. This class defines the inverse transformation (F^{-1} in the paper) that converts
      an HPP sample into our target TPP.
    2. To avoid redundant computations, our base distribution is product of iid unit
      exponential distriubitons. This corresponds to modeling the inter-event times of an HPP.
      We would obtain the HPP arrival times if apply cumulative sum. Since the log-determinant
      of cumulative sum is equalto zero, this doesn't change the results.

    Args:
        transforms (list): List of transformations applied to the distribution.
        burn_in (int): Number of events to discard when computing NLL.

    Implementation inspired by TTPP.
    https://github.com/shchur/triangular-tpp/blob/main/ttpp/flows/transformed_distribution.py

    Implementation inspired by GNTPP.
    https://github.com/BIRD-TAO/GNTPP/blob/main/models/tpp_warper.py
    """

    def __init__(
        self,
        transforms: list[Flow] | Flow,
        burn_in: int = Config.burn_in,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if "t_max_normalization" in kwargs:
            self.t_max_normalization = kwargs["t_max_normalization"]
        if "fixed_normalization" in kwargs:
            self.fixed_normalization = kwargs["fixed_normalization"]
        if isinstance(transforms, Flow):
            self.transforms = nn.ModuleList(
                [
                    transforms,
                ]
            )
        elif isinstance(transforms, list):
            if not all(isinstance(t, Flow) for t in transforms):
                raise ValueError("transforms must be a Flow or a list of Flows")
            self.transforms = nn.ModuleList(transforms)
        else:
            raise ValueError(
                f"transforms must a Flow or a list of Flows, but was {type(transforms)}."
            )
        self.n_transforms = len(self.transforms)
        self.base_dist = StandardExponential(**kwargs)
        self.burn_in = burn_in
        self.n_marks = self.transforms[0].n_marks
        for transform in self.transforms:
            assert (
                self.n_marks == transform.n_marks
            ), f"n_marks of all the layers must match {self.n_marks}, but {transform} has n_marks = {transform.n_marks}."
            if hasattr(transform, "clamp_direction"):
                if hasattr(self, "clamp_direction"):
                    if transform.clamp_direction != self.clamp_direction:
                        self.clamp_direction = "both"
                    else:
                        self.clamp_direction = transform.clamp_direction
                else:
                    self.clamp_direction = transform.clamp_direction

    def update_n_max(self, n_events):
        for transform in self.transforms:
            transform.update_n_max(n_events)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
        cache: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Transforms x by F."""
        if xT is None:
            y = x.clone()
            for transform in self.transforms:
                y, _ = transform.forward(y, k)

            return y

        else:
            if cache is None:
                iter = zip(self.transforms, [None for _ in range(len(self.transforms))])
            else:
                iter = zip(self.transforms, cache[::-1])

            y = x.clone()
            yT = xT.clone()

            for transform, cache in iter:
                y, _, yT, _ = transform.forward(y, k, yT, cache)

            return y, yT

    @torch.jit.export
    def compensator(
        self,
        y: torch.Tensor,
        k: torch.Tensor,
        yT: Optional[torch.Tensor] = None,
        cache_y: bool = False,
    ) -> FlowReturnType:
        """Computes the compensator and the logaritm of the conditional intensity of x given by the inverse of F."""
        x = y.clone()
        log_intensity = torch.zeros_like(y)

        if yT is not None:
            xT = yT.clone()
            log_intensityT = torch.zeros_like(yT)

        if cache_y:
            cache = [y]

        for transform in self.transforms[::-1]:
            if yT is None:
                x, inv_log_det_jac = transform.inverse(x, k)
                log_intensity += inv_log_det_jac
            else:
                x, inv_log_det_jac, xT, inv_log_det_jacT = transform.inverse(x, k, xT)
                log_intensity += inv_log_det_jac
                log_intensityT += inv_log_det_jacT

            if cache_y:
                cache.append(x)

        if cache_y:
            cache = cache[:-1]

        if yT is None:
            if cache_y:
                return x, log_intensity, cache
            else:
                return x, log_intensity
        else:
            if cache_y:
                return x, log_intensity, xT, log_intensityT, cache
            else:
                return x, log_intensity, xT, log_intensityT

    @torch.jit.export
    def inverse(
        self,
        y: torch.Tensor,
        k: torch.Tensor,
        yT: Optional[torch.Tensor] = None,
        cache_y: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Transforms x by the inverse of F."""
        if yT is None:
            if cache_y:
                x, _, cache = self.compensator(y, k, cache_y=cache_y)
                return x, cache
            else:
                x, _ = self.compensator(y, k, cache_y=cache_y)
                return x
        else:
            if cache_y:
                x, _, xT, _, cache = self.compensator(y, k, yT, cache_y=cache_y)
                return x, xT, cache
            else:
                x, _, xT, _ = self.compensator(y, k, yT, cache_y=cache_y)
                return x, xT

    @torch.jit.export
    def log_prob(
        self, y: torch.Tensor, k: torch.Tensor, yT: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of a batch of sequences."""
        # ignore the last event on vector y since it will be accounted for in yT
        B, _ = y.shape
        idxT = (k < self.n_marks).sum(dim=-1) - 1
        _k = k.clone()
        _k[torch.arange(B), idxT] = self.n_marks

        # compute compensator and log of conditional intensity
        x, log_intensity, xT, log_intensityT = self.compensator(y, _k, yT)
        x = x.masked_fill(_k == self.n_marks, 0)
        log_intensity = log_intensity.masked_fill(_k == self.n_marks, 0)

        # compute the survival up until the end of the elapsed time for all event types
        log_survival = self.base_dist.log_survival(x)
        log_survivalT = self.base_dist.log_survival(xT)

        if self.burn_in > 0:
            out = (
                log_intensity[:, self.burn_in :] + log_survival[:, self.burn_in :]
            ).sum(dim=-1)
        else:
            out = (log_intensity + log_survival).sum(dim=-1)

        out += log_survivalT.sum(dim=-1)

        return out

    @torch.jit.export
    def loss(
        self,
        y: torch.Tensor,
        k: torch.Tensor,
        yT: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = self.log_prob(y, k, yT)
        return -log_prob.sum()

    @torch.jit.export
    def expected_event(
        self,
        y: torch.Tensor,
        k: torch.Tensor,
        t_max: float,
        n_samples: int = Config.n_samples,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the expected next event time."""

        B, N = y.shape

        with torch.no_grad():
            # append a dummy event to y to sample from 0 to end inclusive
            _y = F.pad(y, (0, 1))
            _k = F.pad(k, (0, 1), value=self.n_marks)

            # define axis which we will use to integrate
            time_step = t_max / n_samples
            yT_axis = torch.linspace(0, t_max, n_samples, device=y.device)

            yT = F.pad(y, (1, 0)).unsqueeze(0) + yT_axis[:, None, None]
            yT = yT.unsqueeze(-1).expand(-1, -1, -1, self.n_marks)

            # map to compensated space
            _, xT = self.inverse(_y, _k, yT)
            # some of the of the compensators in a multivariate TPP will be
            # running since before the event under consideration. Reset all
            # compensators to the start of the period.
            xT = xT - xT[0]
            xT_ground = xT.sum(dim=-1)
            pdf = self.base_dist.pdf(xT_ground)
            time_step_xT = xT_ground.diff(dim=0)

            # variables needed to mask the final results
            idxT = (k < self.n_marks).sum(dim=-1)
            _k[torch.arange(B), idxT] = 0

            # trapezoidal rule
            cdf = ((pdf[:-1] + pdf[1:]) * time_step_xT).sum(dim=0) / 2
            mean_cdf = cdf[_k != self.n_marks].mean()
            if mean_cdf < 0.6 or mean_cdf > 1.2:
                print(f"Mean CDF accross batches is not close to 1.0: {mean_cdf:.4f}")

            # trapezoidal rule
            y_pred = (yT[:-1, :, :, 0] * (pdf[:-1] + pdf[1:]) * time_step_xT).sum(
                dim=0
            ) / 2
            y_pred = y_pred / cdf
            y_pred[_k == self.n_marks] = 0
            # predict mark as the one with the largest compensator at t_max
            _, k_pred = xT[-1].max(dim=-1)
            k_pred[_k == self.n_marks] = self.n_marks

        return y_pred, k_pred

    @torch.jit.export
    def sample(
        self, y: torch.Tensor, k: torch.Tensor, n_samples: int = Config.n_samples
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw n_samples one-event ahead for each event in history y, k."""

        B, N = y.shape

        with torch.no_grad():
            # append a dummy event to y to sample from 0 to end inclusive
            _y = F.pad(y, (0, 1))
            _k = F.pad(k, (0, 1), value=self.n_marks)

            yT = y.unsqueeze(-1).expand(-1, -1, self.n_marks).unsqueeze(0)
            yT = F.pad(yT, (0, 0, 1, 0))

            # map to compensated space
            # some of the of the compensators in a multivariate TPP will be
            # running since before the event under consideration. therefore,
            # not all condxT will be close to 0 but will carry over since each
            # sub-TPP has fired.
            x, condxT, cache = self.inverse(_y, _k, yT, cache_y=True)
            # conditional sample to ensure xT is at least bigger than condxT.
            xT = self.base_dist.rsample(
                (n_samples, B, N + 1, self.n_marks), cond=condxT, device=_y.device
            )
            _, y_candidates = self.forward(x, _k, xT, cache=cache)

            # due to clamping it is not always the case the inverted
            # conditioned sample is higher than the previous event time; we
            # enforce this condition by taking the maximum between the
            # inverted conditioned sample and the previous event time.
            if hasattr(self, "clamp_direction"):
                y_candidates = torch.maximum(y_candidates, yT)

            y_sample, k_sample = y_candidates.min(dim=-1)

            idxT = (k < self.n_marks).sum(dim=-1)
            _k[torch.arange(B), idxT] = 0
            y_sample[:, _k == self.n_marks] = 0
            k_sample[:, _k == self.n_marks] = self.n_marks

        return y_sample, k_sample

    @torch.jit.export
    def predict_event_time(
        self,
        y_sample: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            y_pred = y_sample.mean(dim=0)

        return y_pred

    @torch.jit.export
    def predict_event_prob(
        self,
        k_sample: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            k_prob = torch.zeros(
                (
                    k_sample.shape[0],
                    k_sample.shape[1],
                    k_sample.shape[2],
                    self.n_marks + 1,
                ),
                device=k_sample.device,
            )
            values = torch.ones(
                (
                    k_sample.shape[0],
                    k_sample.shape[1],
                    k_sample.shape[2],
                    self.n_marks + 1,
                ),
                device=k_sample.device,
            )
            k_prob = k_prob.scatter_add(-1, k_sample.unsqueeze(-1), values)
            k_prob = k_prob.sum(dim=0) / k_sample.shape[0]
            k_prob = k_prob[..., :-1]

        return k_prob

    @torch.jit.export
    def expected_event_prob(
        self, y: torch.Tensor, k: torch.Tensor, t_max: torch.Tensor
    ) -> torch.Tensor:
        B, N = y.shape
        with torch.no_grad():
            # append a dummy event to y to sample from 0 to end inclusive
            _y = F.pad(y, (0, 1))
            _k = F.pad(k, (0, 1), value=self.n_marks)

            yT_axis = torch.tensor([0, t_max], dtype=y.dtype, device=y.device)

            yT = F.pad(y, (1, 0)).unsqueeze(0) + yT_axis[:, None, None]
            yT = yT.unsqueeze(-1).expand(-1, -1, -1, self.n_marks)

            # map to compensated space
            _, xT = self.inverse(_y, _k, yT)
            # some of the of the compensators in a multivariate TPP will be
            # running since before the event under consideration. Reset all
            # compensators to the start of the period.
            xT = xT - xT[0]

            k_prob = F.softmax(xT[-1], dim=-1)

            idxT = (k < self.n_marks).sum(dim=-1)
            _k[torch.arange(B), idxT] = 0
            k_prob[_k == self.n_marks] = 0

        return k_prob

    @torch.jit.export
    def predict_event_type(self, k_prob: torch.Tensor, top: int = 1) -> torch.Tensor:
        k_pred = k_prob.argsort(dim=-1, descending=True)[..., :top]
        k_pred[k_prob.sum(dim=-1) == 0] = self.n_marks
        return k_pred

    @torch.jit.export
    def cumulative_risk_func(self, y, k):
        """Computes the accumulator."""
        return self.inverse(y, k)

    @torch.jit.export
    def intensity_func(
        self,
        y: torch.Tensor,
        k: torch.Tensor,
        dt: float,
        t_max: float,
    ):
        # need to compute the log_intensity like we do in log_prob, then stich it all together
        B, _ = y.shape

        with torch.no_grad():
            n_samples = int(t_max / dt)
            yT = torch.linspace(0, t_max, n_samples, device=y.device)

            intensity = torch.empty(
                (B, n_samples, self.n_marks), dtype=y.dtype, device=y.device
            )

            for i, t in enumerate(yT):
                if i == 0:
                    valid = torch.zeros_like(y).to(bool)
                else:
                    valid = (y < t) * (k < self.n_marks)
                _k = k.masked_fill(~valid, self.n_marks)
                # TODO: ideally we would like to compute the intensity in
                # batch; by simply passing a batch of all yT we want to compute
                # the intensity for
                _yT = t.expand(B, self.n_marks).clone()
                _, _, _, log_intensityT = self.compensator(y, _k, _yT)
                intensity[:, i, :] = log_intensityT.exp()

        return yT, intensity

    @torch.jit.export
    def generate(
        self,
        y: torch.Tensor,
        k: torch.Tensor,
        n_samples: int,
        N_min: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate n_samples ahead starting from history y, k via First Reaction Method (FRM)."""
        B, N = y.shape

        with torch.no_grad():
            if y.nelement() == 0:
                y = torch.zeros((B, 1), dtype=y.dtype, device=y.device)
                k = torch.ones((B, 1), dtype=k.dtype, device=k.device) * self.n_marks
                yT = torch.zeros((B, self.n_marks), dtype=y.dtype, device=y.device)

            idxT = (k < self.n_marks).sum(dim=-1) - 1
            y[:, -1] = y[:, -1].masked_fill((idxT < 0), 0)
            yT = y[torch.arange(B), idxT].unsqueeze(-1).expand(-1, self.n_marks)
            yT = yT.masked_fill((idxT < 0).unsqueeze(-1), 0)
            N = (idxT.max() + 1).item()

            if N < N_min:
                y = F.pad(y, (0, N_min - N - 1), value=0.0)
                k = F.pad(k, (0, N_min - N - 1), value=self.n_marks)

            for _ in range(n_samples):
                x, condxT, cache = self.inverse(y, k, yT, cache_y=True)
                xT = self.base_dist.rsample(
                    (B, self.n_marks), cond=condxT, device=y.device
                )
                _, y_candidates = self.forward(x, k, xT, cache=cache)

                if hasattr(self, "clamp_direction"):
                    y_candidates = torch.maximum(
                        y_candidates, y[torch.arange(B), idxT].unsqueeze(1)
                    )

                y_next, k_next = y_candidates.min(dim=-1)

                if N >= y.shape[1]:
                    y = F.pad(y, (0, 1))
                    k = F.pad(k, (0, 1), value=self.n_marks)

                idxT += 1
                N += 1
                y[torch.arange(B), idxT] = y_next
                k[torch.arange(B), idxT] = k_next

                yT = y_next.unsqueeze(-1).expand(-1, self.n_marks)

            y_out = []
            k_out = []
            for i in range(B):
                y_out.append(
                    y[i, (idxT[i] - n_samples + 1) : (idxT[i] + 1)].unsqueeze(0)
                )
                k_out.append(
                    k[i, (idxT[i] - n_samples + 1) : (idxT[i] + 1)].unsqueeze(0)
                )
            return torch.cat(y_out), torch.cat(k_out)
