from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass(kw_only=True)
class Config:
    dataset: Optional[str] = None
    model_name: Optional[str] = None

    # Output
    checkpoint: Optional[str | Path] = None
    output: Optional[str | Path] = None
    log: Optional[str | Path] = None

    # Data
    n_marks: Optional[int] = None
    n_events: Optional[int] = None

    # Training
    seed: int = 42
    device: int = 0
    batch_size: int = 32
    val_batch_size: int = 16
    learning_rate: float = 1e-3
    max_grad_norm: float = 5.0
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15
    burn_in: int = 0
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    log_every: int = 10

    # Sampling
    n_samples: int = 100

    # Affine
    t_max_normalization: Optional[float] = None
    fixed_normalization: float = 50
    trainable_normalization: float = 1.0
    log_dt_mean_normalization: Optional[float] = None
    log_dt_std_normalization: Optional[float] = None
    dt_max_normalization: Optional[float] = None

    # Block diagonal/RNN/Transformer
    n_blocks: int = 4

    # RNN/Transformer
    n_embd: int = 8

    # Splines
    n_knots: int = 20
    spline_order: int = 2

    # Block diagonal
    block_size: int = 16

    # Transformer
    n_heads: int = 4
    dropout: float = 0.1

    # Debugging
    skip_metrics: bool = False
    enable_nan_check: bool = False
    use_jit: bool = False
