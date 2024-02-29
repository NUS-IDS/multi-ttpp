#!/usr/bin/env python
import argparse
import logging
import torch
import multittpp


def main(config):
    initial_checkpoint = config.checkpoint

    config.output, config.checkpoint, config.log = multittpp.resolve_config_paths(
        config.dataset, config.model_name, config.output
    )

    logger = multittpp.get_logger(__name__, log_path=config.log, level=logging.INFO)

    multittpp.set_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device)
        logger.info(f"CUDA device: {torch.cuda.current_device()}")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    (
        data,
        n_marks,
        t_max_normalization,
        dt_max_normalization,
        N_max,
    ) = multittpp.data.load_dataset(
        dataset=config.dataset,
        fixed_normalization=config.fixed_normalization,
        N_min=config.block_size,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        device=config.device,
    )

    config.n_marks = n_marks
    config.t_max_normalization = t_max_normalization
    config.dt_max_normalization = dt_max_normalization
    config.n_events = N_max

    if initial_checkpoint is None:
        checkpoint_dict = None
        model = getattr(multittpp.models, config.model_name)(**vars(config))
    else:
        checkpoint_dict, model = multittpp.load_model(
            initial_checkpoint, config, logger
        )

    model = model.to(config.device)

    if config.use_jit:
        model = torch.jit.script(model)

    trainer = multittpp.Trainer(
        data=data,
        model=model,
        config=config,
        checkpoint_dict=checkpoint_dict,
    )

    logger.info(f"{config}")

    summary_table, total_params = trainer.count_parameters()
    logger.info(f"Model summary:\n{summary_table}")
    logger.info(f"Total trainable params: {total_params:,d}")

    evaluate_metrics = [
        "LOSS",
        # "NLL",
        # "CE",
        # "MAPE",
        # "RMSE",
        # "CRPS",
        # "TOP1_ACC",
        # "TOP3_ACC",
        "QQDEV",
    ]

    total_epochs = config.epochs

    # trainer.evaluate(
    #     "val",
    #     evaluate_metrics,
    #     f"Metrics: 0 / {total_epochs:d} epochs, val dataset:",
    #     verbose=True,
    # )
    # trainer.evaluate(
    #     "test",
    #     evaluate_metrics,
    #     f"Metrics: 0 / {total_epochs:d} epochs, test dataset:",
    #     verbose=True,
    # )

    trainer.train()

    final_epochs = trainer.checkpoint_dict["epochs"]

    trainer.evaluate(
        "val",
        evaluate_metrics,
        f"Metrics: {final_epochs} / {total_epochs:d} epochs, val dataset:",
        verbose=True,
    )
    trainer.evaluate(
        "test",
        evaluate_metrics,
        f"Metrics: {final_epochs} / {total_epochs:d} epochs, test dataset:",
        verbose=True,
    )

    return trainer


if __name__ == "__main__":
    default_config = multittpp.Config(dataset=None, model_name=None)

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("model_name")

    # Output
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=default_config.checkpoint
    )
    parser.add_argument("-o", "--output", type=str, default=default_config.output)

    # Training
    parser.add_argument("-s", "--seed", default=default_config.seed, type=int)
    parser.add_argument("-d", "--device", default=default_config.device, type=int)
    parser.add_argument(
        "-b", "--batch-size", default=default_config.batch_size, type=int
    )
    parser.add_argument(
        "-vb", "--val-batch-size", default=default_config.val_batch_size, type=int
    )
    parser.add_argument("-e", "--epochs", default=default_config.epochs, type=int)
    parser.add_argument("-p", "--patience", default=default_config.patience, type=int)
    parser.add_argument(
        "-lr", "--learning_rate", default=default_config.learning_rate, type=float
    )
    parser.add_argument(
        "-mg", "--max_grad_norm", default=default_config.max_grad_norm, type=float
    )
    parser.add_argument(
        "-wd", "--weight_decay", default=default_config.weight_decay, type=float
    )
    parser.add_argument("-bi", "--burn-in", default=default_config.burn_in, type=int)
    parser.add_argument(
        "-le", "--log-every", default=default_config.log_every, type=int
    )

    # Affine
    parser.add_argument(
        "-tm",
        "--t-max-normalization",
        default=default_config.t_max_normalization,
        type=float,
    )
    parser.add_argument(
        "-fn",
        "--fixed-normalization",
        default=default_config.fixed_normalization,
        type=float,
    )
    parser.add_argument(
        "-tn",
        "--trainable-normalization",
        default=default_config.trainable_normalization,
        type=float,
    )

    # Block diagonal/RNN/Transformer
    parser.add_argument("-nb", "--n-blocks", default=default_config.n_blocks, type=int)

    # RNN/Transformer
    parser.add_argument("-ne", "--n-embd", default=default_config.n_embd, type=int)

    # Splines
    parser.add_argument("-nk", "--n-knots", default=default_config.n_knots, type=int)
    parser.add_argument(
        "-so", "--spline-order", default=default_config.spline_order, type=int
    )

    # Block diagonal
    parser.add_argument(
        "-bs", "--block-size", default=default_config.block_size, type=int
    )

    # Transformer
    parser.add_argument("-nh", "--n-heads", default=default_config.n_heads, type=int)
    parser.add_argument("-do", "--dropout", default=default_config.dropout, type=float)

    # Debugging
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--enable-nan-check", action="store_true")
    parser.add_argument("--use-jit", action="store_true")

    config = multittpp.Config(**vars(parser.parse_args()))

    main(config)

    print("Done.")
