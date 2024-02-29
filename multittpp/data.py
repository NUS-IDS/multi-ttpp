"""
Implementation inspired by TTPP.
https://github.com/BIRD-TAO/GNTPP/blob/main/datasets/tpp_loader.py
"""
import torch
import pickle
import numpy as np
import torch.utils.data as data_utils
import torch.nn.functional as F

from pathlib import Path
from typing import Optional, TypedDict
from collections.abc import Sequence

from .config import Config


dataset_dir = Path(__file__).parents[1] / "data"


def load_dataset(
    dataset: str,
    fixed_normalization: float = Config.fixed_normalization,
    N_min: int = Config.block_size,
    batch_size: int = Config.batch_size,
    val_batch_size=Config.val_batch_size,
    device=Config.device,
):
    if val_batch_size == None:
        val_batch_size = batch_size

    if isinstance(device, int):
        device = torch.device("cuda", device)

    train_set = SequenceDataset(
        data=open_dataset(dataset_dir, dataset, "train"),
        fixed_normalization=fixed_normalization,
        N_min=N_min,
        device=device,
    )

    validation_set = SequenceDataset(
        data=open_dataset(dataset_dir, dataset, "val"),
        fixed_normalization=fixed_normalization,
        N_min=N_min,
        device=device,
    )

    test_set = SequenceDataset(
        data=open_dataset(dataset_dir, dataset, "test"),
        fixed_normalization=fixed_normalization,
        N_min=N_min,
        device=device,
    )

    t_max_normalization = float(train_set.t_max)
    dt_max_normalization = float(train_set.dt_max)
    N_max = np.max([train_set.N_max, validation_set.N_max, test_set.N_max])

    for d in [train_set, validation_set, test_set]:
        setattr(d, "t_max_normalization", t_max_normalization)
        setattr(d, "dt_max_normalization", dt_max_normalization)
        setattr(d, "N_max", N_max)

    data = {}
    data["train_loader"] = data_utils.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator(device=device),
    )
    data["val_loader"] = data_utils.DataLoader(
        validation_set,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate,
        generator=torch.Generator(device=device),
    )
    data["test_loader"] = data_utils.DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate,
        generator=torch.Generator(device=device),
    )

    assert train_set.n_marks == validation_set.n_marks == test_set.n_marks

    return (
        data,
        train_set.n_marks,
        t_max_normalization,
        dt_max_normalization,
        N_max,
    )


def open_dataset(dataset_dir: str | Path, dataset: str, mode: str):
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
    file = dataset_dir / f"{dataset}/{mode}_manifold_format.pkl"
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def one_hot_embedding(labels: torch.Tensor, n_marks: int) -> torch.Tensor:
    """Embedding labels to one-hot form. Produces an easy-to-use mask to select components of the intensity.

    Args:
        labels (torch.Tensor): class labels, sized [N,].
        n_marks (int): number of classes.

    Returns:
        torch.Tensor: encoded labels, sized [N, n_marks].
    """
    device = labels.device
    y = torch.eye(n_marks).to(device)
    return y[labels]


def collate(dataset_item):
    N_min = dataset_item[0][9]
    device = dataset_item[0][8]
    n_marks = dataset_item[0][7]
    dataset_item = sorted(dataset_item, key=lambda x: len(x[0]), reverse=True)
    in_dts = [item[0] for item in dataset_item]
    out_dts = [item[1] for item in dataset_item]
    in_types = [item[2] for item in dataset_item]
    out_types = [item[3] for item in dataset_item]
    in_times = [item[4] for item in dataset_item]

    seq_lengths = torch.tensor([item[5] for item in dataset_item])
    if (seq_lengths.max() - 1) < N_min:
        short_batch = True
        additional_padding = N_min - (seq_lengths.max() - 1)
    else:
        short_batch = False
        additional_padding = 0
    n_marks = dataset_item[0][7]

    in_dts = torch.nn.utils.rnn.pad_sequence(
        in_dts, batch_first=True, padding_value=0.0
    )
    out_dts = torch.nn.utils.rnn.pad_sequence(
        out_dts, batch_first=True, padding_value=0.0
    )
    in_types = torch.nn.utils.rnn.pad_sequence(
        in_types, batch_first=True, padding_value=n_marks
    )
    out_types = torch.nn.utils.rnn.pad_sequence(
        out_types, batch_first=True, padding_value=n_marks
    )
    in_times = torch.nn.utils.rnn.pad_sequence(
        in_times, batch_first=True, padding_value=0.0
    )

    if short_batch:
        in_dts = F.pad(in_dts, (0, additional_padding), value=0.0)
        out_dts = F.pad(out_dts, (0, additional_padding), value=0.0)
        in_types = F.pad(in_types, (0, additional_padding), value=n_marks)
        out_types = F.pad(out_types, (0, additional_padding), value=n_marks)
        in_times = F.pad(in_times, (0, additional_padding), value=0.0)

    last_times = (
        torch.tensor([item[6] for item in dataset_item], dtype=in_dts.dtype)
        .unsqueeze(-1)
        .expand(-1, n_marks)
    )

    return Batch(
        in_dts.to(device),
        in_types.to(device),
        in_times.to(device),
        seq_lengths.to(device),
        last_times.to(device),
        out_dts.to(device),
        out_types.to(device),
        N_min,
    )


DataType = TypedDict(
    "DataType",
    {
        "timestamps": Sequence[Sequence[float]],
        "types": Sequence[Sequence[int]],
        "lengths": Sequence[int],
        "intervals": Sequence[Sequence[float]],
        "t_max": float,
        "event_type_num": int,
    },
)


class SequenceDataset(data_utils.Dataset):
    """Dataset containing variable length sequences.

    Args:
        data (dict): A dictionary with data and metadata. It contains the
            following items, "timestamps": a sequence of sequence of event
            timestamps, "types": a sequence of sequence of event types, "lengths": a
            sequence of the lenght of the sequence of events, "intervals": a
            sequence of sequence of duration interval between events, "t_max": the
            maximum timestamp in the data, "event_type_num": the number of marks in
            the data.
        batch_size (int): The batch size which is passed to the data loader.
        device (str or torch.device, optional): The selected device.
        scale_normalization (float): The maximum timestamp after scale normalization.
    """

    t_max_normalization: float
    dt_max_normalization: float
    N_max: int

    def __init__(
        self,
        data: DataType,
        fixed_normalization: float = Config.fixed_normalization,
        N_min: int = Config.block_size,
        device: Optional[int | str | torch.device] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.fixed_normalization = fixed_normalization
        self.N_min = N_min
        self.data = data
        self.process_data()

    def process_data(self):
        (
            self.seq_times,
            self.seq_types,
            self.seq_lengths,
            self.seq_dts,
            self.n_marks,
        ) = (
            self.data["timestamps"],
            self.data["types"],
            self.data["lengths"],
            self.data["intervals"],
            self.data["event_type_num"],
        )

        # recompute t_max to ensure it is consistent with timestamps
        self.t_max = np.concatenate(self.seq_times).max()
        self.dt_max = np.concatenate(self.seq_dts).max()

        self.seq_lengths = torch.tensor(self.seq_lengths)
        self.N_max = self.seq_lengths.max().item()

        # remove the last point of each sequence for estimation purposes
        self.in_times = [torch.tensor(t[:-1]).float() for t in self.seq_times]
        self.out_times = [torch.tensor(t[1:]).float() for t in self.seq_times]

        self.in_dts = [torch.tensor(dt[:-1]).float() for dt in self.seq_dts]
        self.out_dts = [torch.tensor(dt[1:]).float() for dt in self.seq_dts]

        self.in_types = [torch.tensor(m[:-1]).long() for m in self.seq_types]
        self.out_types = [torch.tensor(m[1:]).long() for m in self.seq_types]

        self.last_times = torch.tensor([t[-2] for t in self.seq_times])

        self.validate_times()

    @property
    def num_series(self):
        return len(self.in_times)

    def get_dt_statistics(self):
        flat_in_dts_log = (torch.cat(self.in_dts) + 1e-8).log()
        return (
            flat_in_dts_log.mean(),
            flat_in_dts_log.std(),
            flat_in_dts_log.exp().max(),
        )

    def validate_times(self):
        if len(self.in_times) != len(self.out_times):
            raise ValueError("in_times and out_times have different lengths.")

        for s1, s2, s3, s4 in zip(
            self.in_times, self.out_times, self.in_types, self.out_types
        ):
            if len(s1) != len(s2) or len(s3) != len(s4):
                raise ValueError("Some in/out series have different lengths.")
            if s3.max() >= self.n_marks or s4.max() >= self.n_marks:
                raise ValueError("Marks should not be larger than number of classes.")

    def scale(self):
        """Scale each sequence by `self.fixed_normalization / self.t_max_normalization`."""
        self.in_times = [
            t / self.t_max_normalization * self.fixed_normalization
            for t in self.in_times
        ]
        self.in_dts = [
            t / self.t_max_normalization * self.fixed_normalization for t in self.in_dts
        ]

        self.out_times = [
            t / self.t_max_normalization * self.fixed_normalization
            for t in self.out_times
        ]
        self.out_dts = [
            t / self.t_max_normalization * self.fixed_normalization
            for t in self.out_dts
        ]

    def __getitem__(self, key):
        return (
            self.in_dts[key],
            self.out_dts[key],
            self.in_types[key],
            self.out_types[key],
            self.in_times[key],
            self.seq_lengths[key],
            self.last_times[key],
            self.n_marks,
            self.device,
            self.N_min,
        )

    def __len__(self):
        return self.num_series

    def __repr__(self):
        return f"SequenceDataset({self.num_series})"


class Batch:
    def __init__(
        self,
        in_dts: torch.Tensor,
        in_types: torch.Tensor,
        in_times: torch.Tensor,
        seq_lengths: torch.Tensor,
        last_times: torch.Tensor,
        out_dts: torch.Tensor,
        out_types: torch.Tensor,
        N_min: int,
    ) -> None:
        self.in_dts = in_dts
        self.in_types = in_types.long()
        self.in_times = in_times
        self.seq_lengths = seq_lengths
        self.last_times = last_times
        self.out_dts = out_dts
        self.out_types = out_types.long()
        self.N_min = N_min
