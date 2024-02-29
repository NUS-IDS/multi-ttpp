#!/usr/bin/env python
import pickle
import argparse
import logging
import torch
import multittpp

from pathlib import Path
from time import time_ns
from tqdm import tqdm


def main(
    checkpoint: str | Path, device: int, val_batch_size: int, log_save: bool = False
):
    if isinstance(checkpoint, str):
        checkpoint = Path(checkpoint)
    if log_save:
        log_path = checkpoint.parent / f"{checkpoint.stem}-benchmark.log"
        pickle_path = checkpoint.parent / f"{checkpoint.stem}-benchmark.pkl"
    else:
        log_path = None
    logger = multittpp.get_logger(__name__, log_path=log_path, level=logging.INFO)
    trainer = multittpp.Trainer.load_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        val_batch_size=val_batch_size,
        log_path=log_path,
    )

    n_iter = 10
    n_samples = [25, 50, 100, 200, 400, 800, 1600]
    times = [[] for _ in n_samples]

    for t, i, batch in trainer.batch(
        dataset="test", grad=False, verbose=True, total=n_iter
    ):
        if i == n_iter:
            break

        for j, n in enumerate(tqdm(n_samples, leave=False)):
            start = time_ns()
            y_gen, k_gen = trainer.generate(
                batch, start_ix=1, n_samples=n, N_min=batch.N_min
            )
            elapsed = time_ns() - start
            times[j].append(elapsed)

    if log_save:
        with open(pickle_path, "wb") as f:
            pickle.dump({"n_samples": n_samples, "times": times}, f)


if __name__ == "__main__":
    default_config = multittpp.Config(dataset=None, model_name=None)

    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint", type=str)
    parser.add_argument("-d", "--device", default=default_config.device, type=int)
    parser.add_argument(
        "-vb", "--val-batch-size", default=default_config.val_batch_size, type=int
    )
    parser.add_argument("--no-log-save", action="store_false", dest="log_save")

    args = parser.parse_args()

    main(**vars(args))
