import torch
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Optional


def plot_intensity(
    bix: int,
    intensity_pred: torch.Tensor,
    yT_pred: torch.Tensor,
    kix: Optional[int | list[int]] = None,
    intensity: Optional[torch.Tensor] = None,
    yT: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    title: Optional[str] = "Intensity rate",
    y_label_extra: Optional[str] = None,
    show: bool = True,
    ax: Optional[mpl.axes.Axes] = None,
):
    if yT_pred.shape[0] != intensity_pred.shape[1]:
        raise ValueError(
            "yT_pred and intensity_pred must have the same number of samples."
        )

    if kix is None:
        kix = list(range(intensity_pred.shape[-1]))

    if isinstance(kix, int):
        kix = [kix]

    intensity_pred = intensity_pred[bix]

    if not ((intensity is None) and (yT is None)):
        if (intensity is not None) and (yT is not None):
            if intensity.shape[1] != yT.shape[0]:
                raise ValueError(
                    "yT and intensity must have the same number of samples."
                )
            intensity = intensity[bix]
        else:
            raise ValueError("You must pass both intensity and yT.")

    if not ((y is None) and (k is None)):
        if (y is not None) and (k is not None):
            y = y[bix]
            k = k[bix]
        else:
            raise ValueError("You must pass both y and k.")

    if t_min is not None or t_max is not None:
        t_mask_pred = torch.ones_like(yT_pred, dtype=bool)

    if yT is not None:
        if t_min is not None or t_max is not None:
            t_mask = torch.ones_like(yT, dtype=bool)

    if y is not None:
        t_y_mask = torch.ones_like(y, dtype=bool)

    if t_min is not None:
        t_mask_pred = t_mask_pred & (yT_pred >= t_min)
        if yT is not None:
            t_mask = t_mask & (yT >= t_min)
        if y is not None:
            t_y_mask = t_y_mask & (y >= t_min)

    if t_max is not None:
        t_mask_pred = t_mask_pred & (yT_pred <= t_max)
        if yT is not None:
            t_mask = t_mask & (yT <= t_max)
        if y is not None:
            t_y_mask = t_y_mask & (y <= t_max)

    if t_min is not None or t_max is not None:
        yT_pred = yT_pred[t_mask_pred]
        intensity_pred = intensity_pred[t_mask_pred]
        if yT is not None:
            yT = yT_pred[t_mask]
            intensity = intensity[t_mask]
        if y is not None:
            y = y[t_y_mask]
            k = k[t_y_mask]

    if isinstance(k, int):
        k = [k]

    if ax is None:
        figsize = (12, 4 * len(kix))
        fig, ax = plt.subplots(len(kix), 1, sharex=True, sharey=True, figsize=figsize)
    else:
        fig = plt.gcf()
        figsize = fig.get_size_inches()

    if len(kix) == 1:
        ax = [ax]

    yT_pred = yT_pred.cpu().numpy()
    if yT is not None:
        yT = yT.cpu().numpy()
    if y is not None:
        y = y.cpu().numpy()
        k = k.cpu().numpy()

    if figsize[0] <= 2 or figsize[1] <= 2:
        linewidth = 0.5
        marker_size = 3
    else:
        linewidth = 1
        marker_size = 6

    for i, ix in enumerate(kix):
        if y_label_extra is None:
            ax[i].set_ylabel(f"$\lambda_{ix+1} (t)$")
        else:
            ax[i].set_ylabel(f"{y_label_extra} $\lambda_{ix+1} (t)$")
        _intensity_pred = intensity_pred[:, ix].cpu().numpy()
        ax[i].plot(yT_pred, _intensity_pred, label="predicted", linewidth=linewidth)
        color = ax[i].lines[-1].get_color()
        if yT is not None:
            _intensity = intensity[:, ix].cpu().numpy()
            ax[i].plot(yT, _intensity, label="true", linewidth=linewidth)
        if y is not None:
            _y = y[k == ix]
            ax[i].scatter(_y, np.repeat(0, _y.shape[0]), marker="^", s=marker_size)

    ax[0].legend(loc="upper left")
    ax[-1].set_xlabel(f"$t$")

    if title is not None:
        plt.suptitle(title)

    if show is True:
        plt.show()

    return ax[0].figure


def plot_qq(
    quant_pred: list[torch.Tensor],
    probs: torch.Tensor,
    prob_axis: bool = True,
    title: Optional[str] = "QQ-Plot",
    y_label_extra: Optional[str] = None,
    show: bool = True,
    rasterized: bool = False,
    ax: Optional[mpl.axes.Axes] = None,
):
    if prob_axis:
        quant_max = (
            torch.tensor([(1 - (-q.max()).exp()) for q in quant_pred if len(q) != 0])
            .max()
            .item()
        )
        quant_max = max(quant_max, 1)
    else:
        quant_max = (
            torch.tensor([q.max() for q in quant_pred if len(q) != 0]).max().item()
        )

    if ax is None:
        figsize = (6, 6)
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)
    else:
        fig = plt.gcf()
        figsize = fig.get_size_inches()

    ax.axline(
        [0, 0], [quant_max, quant_max], color="lightgray", linestyle="-", zorder=99
    )

    if prob_axis:
        ax.set_xlabel("Expected prob.")
        if y_label_extra:
            ax.set_ylabel(f"{y_label_extra}\nEmpirical prob.")
        else:
            ax.set_ylabel("Empirical prob.")
    else:
        ax.set_xlabel("Expected quant.")
        if y_label_extra:
            ax.set_ylabel(f"{y_label_extra}\nEmpirical quant.")
        else:
            ax.set_ylabel("Empirical quant.")

    ax.set_xlim([0, quant_max])
    ax.set_ylim([0, quant_max])
    ax.axis("square")

    if prob_axis:
        ticks = [0.1, 0.3, 0.5, 0.7, 0.9]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t*100:.0f}%" for t in ticks], rotation=90)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t*100:.0f}%" for t in ticks])
    else:
        ticks = [0, 1, 2, 3, 4]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    if prob_axis:
        expected = probs.cpu().numpy()
    else:
        expected = -(1 - probs).log().cpu().numpy()

    if figsize[0] <= 2 or figsize[1] <= 2:
        marker_size = 0.5
    elif len(quant_pred) > 5:
        marker_size = 2
    else:
        marker_size = 3

    if len(quant_pred) == 1:
        color = "red"
    else:
        color = None

    for quant in quant_pred:
        if len(quant) == 0:
            continue
        if prob_axis:
            obtained = (1 - (-quant).exp()).cpu().numpy()
        else:
            obtained = quant.cpu().numpy()
        ax.scatter(expected, obtained, s=marker_size, color=color, rasterized=rasterized)

    if title is not None:
        ax.set_title(title)

    if show is True:
        plt.show()

    return ax.figure
