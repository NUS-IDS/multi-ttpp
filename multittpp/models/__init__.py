"""
All models are implemented like normalizing flow models:
We specify TPP densities by defining the sequence of transformations that
map an arbitrary TPP sample into a vector, where each component follows
unit exponential distribution (which corresponds to the inter-event times
of a homogeneous Poisson process with unit rate).

Implementation inspired by TTPP.
https://github.com/shchur/triangular-tpp/blob/main/ttpp/models.py
"""

from .. import flows
from .base import TransformedExponential
from ..config import Config

__all__ = [
    # TTPP
    "InhomogeneousPoisson",
    "Renewal",
    "ModulatedRenewal",
    "SplineRNN",
    "TriTPP",
    "SplineTransformer",
    "InvertibleTransformer",
    "GPT",
    # Benchmarks
    "OmiRNN",
    "SAHP",
]

#
# TTPP
#


def trainable_interarrival_time(
    *,
    n_marks,
    n_knots,
    trainable_normalization,
    fixed_normalization,
    t_max_normalization,
    **kwargs,
):
    return [
        # interarrival time
        flows.CumSum(n_marks=n_marks),
        # scale by trainable_normalization
        flows.Affine(
            n_marks=n_marks,
            scale_init=trainable_normalization,
            trainable=True,
        ),
        # spline on square box [0, fixed_normalization]
        flows.Spline(
            n_marks=n_marks,
            n_knots=n_knots,
            right=fixed_normalization,
            top=fixed_normalization,
            **kwargs,
        ),
        # scale to fixed_normalization / t_max_normalization
        flows.Affine(
            n_marks=n_marks,
            scale_init=t_max_normalization / fixed_normalization,
            trainable=False,
        ),
    ]


class InhomogeneousPoisson(TransformedExponential):
    """Inhomogeneous Poisson process defined on the interval."""

    def __init__(
        self,
        *,
        n_marks: int,
        n_knots: int = Config.n_knots,
        t_max_normalization: float,
        fixed_normalization: float = Config.fixed_normalization,
        trainable_normalization: float = Config.trainable_normalization,
        burn_in: int = Config.burn_in,
        **kwargs,
    ):
        transforms = trainable_interarrival_time(
            n_marks=n_marks,
            n_knots=n_knots,
            trainable_normalization=trainable_normalization,
            fixed_normalization=fixed_normalization,
            t_max_normalization=t_max_normalization,
            **kwargs,
        )
        super().__init__(transforms, burn_in)


class Renewal(TransformedExponential):
    """Renewal process, all inter-event times are sampled i.i.d."""

    def __init__(
        self,
        *,
        n_marks: int,
        n_knots: int = Config.n_knots,
        t_max_normalization: float,
        fixed_normalization: float = Config.fixed_normalization,
        trainable_normalization: float = Config.trainable_normalization,
        burn_in: int = Config.burn_in,
        **kwargs,
    ):
        transforms = [
            flows.ExpNegative(n_marks=n_marks),
            flows.Spline(n_marks=n_marks, n_knots=n_knots, **kwargs),
            flows.NegativeLog(n_marks=n_marks),
            # scale by trainable_normalization
            flows.Affine(
                n_marks=n_marks,
                scale_init=trainable_normalization,
                trainable=True,
            ),
            # interarrival time
            flows.CumSum(n_marks=n_marks),
            # scale to fixed_normalization / t_max_normalization
            flows.Affine(
                n_marks=n_marks,
                scale_init=t_max_normalization / fixed_normalization,
                trainable=False,
            ),
        ]
        super().__init__(transforms, burn_in)


class ModulatedRenewal(TransformedExponential):
    """Modulated renewal process - generalized Renewal and InhomogeneousPoisson."""

    def __init__(
        self,
        *,
        n_marks: int,
        n_knots: int = Config.n_knots,
        t_max_normalization: float,
        fixed_normalization: float = Config.fixed_normalization,
        trainable_normalization: float = 1.0,
        burn_in: int = Config.burn_in,
        **kwargs,
    ):
        transforms = [
            flows.ExpNegative(n_marks=n_marks),
            flows.Spline(n_marks=n_marks, n_knots=n_knots, **kwargs),
            flows.NegativeLog(n_marks=n_marks),
        ] + trainable_interarrival_time(
            n_marks=n_marks,
            n_knots=n_knots,
            trainable_normalization=trainable_normalization,
            fixed_normalization=fixed_normalization,
            t_max_normalization=t_max_normalization,
            **kwargs,
        )
        super().__init__(transforms, burn_in)


def triangular_layers(n_marks, n_blocks, block_size=Config.block_size, **kwargs):
    """Block-diagonal layers used in the TriTPP model."""
    result = []
    for i in range(n_blocks):
        offset = block_size // 2 * (i % 2)
        result.append(
            flows.BlockDiagonal(
                n_marks=n_marks, block_size=block_size, offset=offset, **kwargs
            )
        )
    return result


class TriTPP(TransformedExponential):
    """TriTPP model with learnable block-diagonal layers, generalized Modulated Renewal."""

    def __init__(
        self,
        *,
        n_marks: int,
        n_blocks: int = Config.n_blocks,
        n_knots: int = Config.n_knots,
        t_max_normalization: float,
        fixed_normalization: float = Config.fixed_normalization,
        trainable_normalization: float = Config.trainable_normalization,
        burn_in: int = Config.burn_in,
        **kwargs,
    ):
        transforms = (
            [
                flows.ExpNegative(n_marks=n_marks),
                flows.Spline(n_marks=n_marks, n_knots=n_knots, **kwargs),
                flows.Logit(n_marks=n_marks),
            ]
            + triangular_layers(n_marks=n_marks, n_blocks=n_blocks, **kwargs)
            + [
                flows.Sigmoid(n_marks=n_marks),
                flows.Spline(n_marks=n_marks, n_knots=n_knots, **kwargs),
                flows.NegativeLog(n_marks=n_marks),
            ]
            + trainable_interarrival_time(
                n_marks=n_marks,
                n_knots=n_knots,
                trainable_normalization=trainable_normalization,
                fixed_normalization=fixed_normalization,
                t_max_normalization=t_max_normalization,
                **kwargs,
            )
        )
        super().__init__(transforms, burn_in)


class SplineTransformer(TransformedExponential):
    """Transformer-based autoregressive model with splines."""

    def __init__(
        self,
        *,
        n_marks: int,
        n_events: int,
        n_knots: int = Config.n_knots,
        t_max_normalization: float,
        fixed_normalization: float = Config.fixed_normalization,
        trainable_normalization: float = Config.trainable_normalization,
        burn_in: int = Config.burn_in,
        **kwargs,
    ):
        transforms = [
            flows.ExpNegative(n_marks=n_marks),
            flows.SplineTransformer(
                n_marks=n_marks, n_events=n_events, n_knots=n_knots, **kwargs
            ),
            flows.NegativeLog(n_marks=n_marks),
        ] + trainable_interarrival_time(
            n_marks=n_marks,
            n_knots=n_knots,
            trainable_normalization=trainable_normalization,
            fixed_normalization=fixed_normalization,
            t_max_normalization=t_max_normalization,
            **kwargs,
        )
        super().__init__(transforms, burn_in)
