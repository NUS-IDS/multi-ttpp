#!/usr/bin/env python
import argparse
import logging
import torch
import multittpp

from pathlib import Path


def main(
    checkpoint: str | Path, device: int, val_batch_size: int, log_save: bool = False
):
    if isinstance(checkpoint, str):
        checkpoint = Path(checkpoint)
    if log_save:
        log_path = checkpoint.parent / f"{checkpoint.stem}-evaluation.log"
    else:
        log_path = None
    logger = multittpp.get_logger(__name__, log_path=log_path, level=logging.INFO)
    trainer = multittpp.Trainer.load_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        val_batch_size=val_batch_size,
        log_path=log_path,
    )

    trainer.config.n_samples = 100

    trainer.evaluate(
        "val",
        ["LOSS", "QQDEV"],
        verbose=True,
    )

    evaluate_metrics = [
        "LOSS",
        "NLL",
        "CE_COND",
        "CE_SAMPLED_NONCOND",
        "MAPE_SAMPLED",
        "RMSE_SAMPLED",
        "MAPE_EXPECTED",
        "RMSE_EXPECTED",
        "CRPS",
        "TOP1_ACC_COND",
        "TOP3_ACC_COND",
        "TOP1_ACC_SAMPLED_NONCOND",
        "TOP3_ACC_SAMPLED_NONCOND",
        "TOP1_ACC_EXPECTED_NONCOND",
        "TOP3_ACC_EXPECTED_NONCOND",
        "QQDEV",
    ]

    trainer.evaluate(
        "test",
        evaluate_metrics,
        verbose=True,
    )


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
