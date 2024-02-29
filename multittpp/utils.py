import subprocess
import sys
import logging
import time
import numpy as np

import torch
import torch.nn.functional as F

from pathlib import Path
from typing import Callable, Optional

from . import models
from .config import Config


def get_logger(name: str, log_path: Optional[str | Path] = None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if name == "__main__":

        def logger_exception_hook(exc_type, exc_value, exc_traceback):
            logger.error("", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = logger_exception_hook
    if log_path is not None:
        if isinstance(log_path, str):
            log_path = Path(log_path)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
        )
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Add console handler.
    console_formatter = logging.Formatter(
        "\033[1m%(asctime)s - %(name)s - %(levelname)s:\033[1;0m %(message)s"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    if log_path is not None:
        logger.info(f"Log path: {log_path}")
    return logger


def remove_file_handlers(logger):
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            logger.removeHandler(handler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model(
    checkpoint_path: Path, config: Config, logger: logging.Logger
) -> tuple[dict, models.TransformedExponential]:
    logger.info(f"Loading from checkpoint path '{checkpoint_path}'.")
    if torch.cuda.is_available():
        checkpoint_dict = torch.load(checkpoint_path, map_location="cuda")
    else:
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model = getattr(models, checkpoint_dict["config"]["model_name"])(**vars(config))
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    return checkpoint_dict, model


def resolve_config_paths(
    dataset: str, model_name: str, output: Optional[str | Path] = None
) -> tuple[Optional[str | Path], Optional[str | Path], Optional[str | Path]]:
    if output is None:
        return None, None, None

    if isinstance(output, str):
        output = Path(output)

    assert output.is_dir()

    output = output.resolve()

    try:
        git_hash = (
            subprocess.check_output(
                ["git", "describe", "--always", "--dirty"],
                cwd=Path(__file__).resolve().parent,
            )
            .strip()
            .decode()
        )
        git_hash = f"-{git_hash}"
    except subprocess.CalledProcessError:
        git_hash = ""

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    stem = f"{model_name}-{dataset}-{timestamp}{git_hash}"

    checkpoint = output / f"{stem}.pth"

    log = output / f"{stem}.log"

    return output, checkpoint, log


def hawkes_intensity(
    y: torch.Tensor,
    k: torch.Tensor,
    n_marks: int,
    dt: float,
    t_max: float,
    adj: torch.Tensor,
    baselines: torch.Tensor,
    kernels: list[list[Callable[[torch.Tensor], torch.Tensor]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the conditional intensity of a Multivariate Hawkes process at regular intervals.

    .. math::

        \lambda_i (t) = \mu_i + \sum_{j=1}^{n_marks} a_{i,j} \sum_{k: t_{j,k} < t} \phi_{i,j} (t - t_{j,k})

    Args:
      y (torch.Tensor): Events, (B, N).
      k (torch.Tensor): Marks, (B, N).
      n_marks (int): The total number of marks.
      dt (float): The desired interval duration between each evaluation.
      t_max (float): The maximum time.
      adj (torch.Tensor): The adjacency matrix :math:`a_{i,j}`, (n_marks, n_marks).
      baselines (torch.Tensor): The baseline rates :math:`\mu_i`, (n_marks).
      kernels (list[list[lambda]]): A nested list of kernel functions such that
          `kernels[i][j]` corresponds to :math:`\phi_{i,j}`.

    Returns:
        torch.Tensor: Evaluation times, (int(t_max) / dt).
        torch.Tensor: Intensity rates, (B, int(t_max) / dt, n_marks).
    """
    B, _ = y.shape

    with torch.no_grad():
        n_samples = int(t_max / dt)
        yT = torch.linspace(dt, t_max, n_samples - 1, device=y.device)
        _y = (yT[:, None, None] - y) * (y < yT[:, None, None]) * (k < n_marks)

        intensity = baselines[None, None, :].expand(B, n_samples - 1, -1).clone()

        for i in range(n_marks):
            for j in range(n_marks):
                if adj[i, j] == 0:
                    continue
                kernel = kernels[i][j]
                intensity[:, :, i] += (
                    (kernel(_y) * (y < yT[:, None, None]) * (k == j))
                    .sum(-1)
                    .transpose(1, 0)
                )

        yT = F.pad(yT, (1, 0))
        intensity = F.pad(intensity, (0, 0, 1, 0))
        intensity[:, 0, :] = baselines

    return yT, intensity
