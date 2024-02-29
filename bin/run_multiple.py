#!/usr/bin/env python
import pynvml as pynvml
import subprocess
import psutil
import time
import multittpp
import re
import dataclasses
import pandas as pd
import warnings

from tqdm import tqdm
from pathlib import Path
from typing import Optional


def check_gpu_usage(
    min_memory: Optional[int] = 5,
    gpus: Optional[int | list] = None,
    process_exceptions: Optional[str | list] = None,
    user_exceptions: Optional[str | list[str]] = None,
):
    """List free GPU resources.

    Args:
      gpus (int or list[int], optional): Only consider the listed GPUs.
      process_exceptions (str or list[str], optional): Processes whose names do
        not belong to the listed process names are considered free.
      user_exceptions (str or list[str], optional): Processes that do not
        belong to the listed users are considered free.
      min_memory (int): Minimum amount of free memory

    Implementation inspired by Github Gist.
    https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    """
    pynvml.nvmlInit()
    # print ("Driver Version:", pynvml.nvmlSystemGetDriverVersion())
    deviceCount = pynvml.nvmlDeviceGetCount()
    free_gpus = []
    if isinstance(gpus, int):
        gpus = [gpus]
    for i in range(deviceCount):
        if (gpus is not None) and (i not in gpus):
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem.free / (1024**3)
        if free_memory is not None and free_memory < min_memory:
            continue

        free = True
        if (process_exceptions is not None) or (user_exceptions is not None):
            if isinstance(process_exceptions, str):
                process_exceptions = [process_exceptions]
            if isinstance(user_exceptions, str):
                user_exceptions = [user_exceptions]
            procs = [
                *pynvml.nvmlDeviceGetComputeRunningProcesses(handle),
                *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle),
            ]
            for p in procs:
                try:
                    process = psutil.Process(p.pid)
                except psutil.NoSuchProcess:
                    continue

                if (process_exceptions is not None) and (user_exceptions is not None):
                    if (process.name in process_exceptions) and (
                        process.username() in user_exceptions
                    ):
                        free = False
                        break
                elif (
                    (process_exceptions is not None)
                    and (process.name in process_exceptions)
                ) or (
                    (user_exceptions is not None)
                    and (process.username() in user_exceptions)
                ):
                    free = False
                    break
        if free:
            free_gpus.append(i)

    pynvml.nvmlShutdown()

    print(f"Free GPUs: [{','.join(map(str, free_gpus))}]")

    return free_gpus


def get_all_runs(reset=False):
    names = []
    ignore_names = [
        "checkpoint",
        "output",
        "log",
        "device",
        "val_batch_size",
        "skip_metrics",
        "enable_nan_check",
        "use_jit",
        "epochs",
    ]
    names.append("ckp")
    train_names = [
        "best_train_loss",
        "best_val_loss",
        "best_test_loss",
        "train_losses",
        "val_losses",
        "epochs",
        "duration",
        "iter_per_second",
    ]
    names += train_names
    for field in dataclasses.fields(multittpp.Config):
        if field.name in ignore_names:
            continue
        names.append(field.name)
    min_memory = 5
    free_gpus = check_gpu_usage(min_memory=min_memory)
    while len(free_gpus) == 0:
        print("No GPUs available, sleeping.")
        time.sleep(120)
        free_gpus = check_gpu_usage(min_memory=min_memory)
    device = free_gpus[0]
    previous_runs_path = Path("./checkpoints/runs.pkl")
    if not reset and previous_runs_path.exists():
        previous_runs = pd.read_pickle(previous_runs_path)
    else:
        previous_runs = pd.DataFrame(columns=names)
    previous_ckps = set(previous_runs.ckp.values)
    runs = []
    total_ckps = len([p for p in Path("./checkpoints").glob("*.pth")])
    for ckp in tqdm(Path("./checkpoints").glob("*.pth"), total=total_ckps):
        if ckp in previous_ckps:
            continue
        trainer = multittpp.Trainer.load_from_checkpoint(
            checkpoint=ckp,
            device=device,
            log=False,
        )
        run = {"ckp": ckp}
        for k, v in dataclasses.asdict(trainer.config).items():
            if k in names:
                run[k] = v
        for k in train_names:
            run[k] = trainer.checkpoint_dict[k]
        qqdev = trainer.evaluate("val", ["QQDEV"])["QQDEV"]
        run["best_qqdev_val"] = qqdev
        train_time = re.search(r"\w+-(2023\d{4}-\d{6})", str(ckp))
        run["train_time"] = pd.to_datetime(train_time.group(1))
        runs.append(run)
    runs = pd.DataFrame.from_dict(runs)
    runs = pd.concat([previous_runs, runs])
    runs.to_pickle(previous_runs_path)
    return runs


def is_config_trained(runs, **fields):
    query = runs.copy()
    # variables that do not affect certain models should not be consulted
    if "model_name" in fields:
        if fields["model_name"] == "SplineTransformer":
            if "block_size" in fields:
                del fields["block_size"]
        if fields["model_name"] in ["TriTPP", "ModulatedRenewal"]:
            if "n_embd" in fields:
                del fields["n_embd"]
    for k, v in fields.items():
        query = query[query[k] == v]
    if len(query) > 0:
        return True
    else:
        return False


def filter_runs(runs, **fields):
    query = runs.copy()
    for k, v in fields.items():
        query = query[query[k] == v]
    return query


def remove_runs(runs, to_remove):
    new_runs = runs.copy()
    for i, run in to_remove.iterrows():
        seed = run["seed"]
        learning_rate = run["learning_rate"]
        n_knots = run["n_knots"]
        n_embd = run["n_embd"]
        block_size = run["block_size"]
        n_blocks = run["n_blocks"]
        model_name = run["model_name"]
        dataset = run["dataset"]
        subset = filter_runs(
            runs,
            seed=seed,
            learning_rate=learning_rate,
            n_knots=n_knots,
            n_embd=n_embd,
            block_size=block_size,
            n_blocks=n_blocks,
            model_name=model_name,
            dataset=dataset,
        )
        new_runs = new_runs[~new_runs.ckp.isin(subset.ckp)]
    return new_runs


def multiple_experiment(
    runs, min_memory=16, min_memory_transformer=30, gpus=None, dry_run=False
):
    """Run multiple experiments."""
    for seed in [42]:
        for learning_rate in [1e-3]:
            for n_knots in [10, 20, 50]:
                for block_size in [8, 16, 32]:
                    for n_blocks in [2, 4]:
                        for n_embd in [8, 16]:
                            for model_name in [
                                "TriTPP",
                                "ModulatedRenewal",
                                "SplineTransformer",
                            ]:
                                for dataset in [
                                    "mimic",
                                    "mooc",
                                    "retweet",
                                    "simulated",
                                    "stackoverflow",
                                    "yelp",
                                ]:
                                    # variables that do not affect certain models do not need permutation
                                    if (model_name == "SplineTransformer") and (
                                        block_size in [8, 32]
                                    ):
                                        continue
                                    if (
                                        model_name in ["TriTPP", "ModulatedRenewal"]
                                    ) and (
                                        n_embd
                                        in [
                                            16,
                                        ]
                                    ):
                                        continue
                                    # Spline Transformer is already large, we want to avoid excessively large models
                                    if (model_name == "SplineTransformer") and (
                                        n_knots
                                        in [
                                            50,
                                        ]
                                    ):
                                        continue
                                    if model_name == "SplineTransformer":
                                        _min_memory = min_memory_transformer
                                    else:
                                        _min_memory = min_memory
                                    if (model_name == "SplineTransformer") and (
                                        dataset in ["mooc", "yelp"]
                                    ):
                                        b = 10
                                        vb = 10
                                    else:
                                        b = 50
                                        vb = 50
                                    if is_config_trained(
                                        runs,
                                        learning_rate=learning_rate,
                                        n_knots=n_knots,
                                        n_embd=n_embd,
                                        block_size=block_size,
                                        n_blocks=n_blocks,
                                        model_name=model_name,
                                        dataset=dataset,
                                        seed=seed,
                                    ):
                                        continue
                                    print(
                                        f"Training {dataset}, {model_name}, n_blocks={n_blocks}, block_size={block_size}, n_embd={n_embd}, n_knots={n_knots}, learning_rate={learning_rate}"
                                    )
                                    if not dry_run:
                                        free_gpus = check_gpu_usage(
                                            gpus=gpus, min_memory=_min_memory
                                        )
                                        while len(free_gpus) == 0:
                                            print("No GPUs available, sleeping.")
                                            time.sleep(120)
                                            free_gpus = check_gpu_usage(
                                                gpus=gpus, min_memory=_min_memory
                                            )
                                        gpu = free_gpus[0]
                                        expname = f"/tmp/exp{gpu}-train-{dataset}-{model_name}-nb{n_blocks}-bs{block_size}-nk{n_knots}-ne{n_embd}"
                                        print(f"Launching {expname}.")
                                        subprocess.run(
                                            [
                                                "dtach",
                                                "-n",
                                                expname,
                                                "python",
                                                "./bin/experiment.py",
                                                dataset,
                                                model_name,
                                                "-d",
                                                f"{gpu:d}",
                                                "-lr",
                                                f"{learning_rate:E}",
                                                "-nb",
                                                f"{n_blocks}",
                                                "-bs",
                                                f"{block_size}",
                                                "-nk",
                                                f"{n_knots}",
                                                "-ne",
                                                f"{n_embd}",
                                                "-e",
                                                "1_000",
                                                "-b",
                                                f"{b}",
                                                "-vb",
                                                f"{vb}",
                                                "-o",
                                                "checkpoints",
                                            ]
                                        )
                                        time.sleep(10)


def select_best_model(runs):
    best_runs = []

    for model_name in ["TriTPP", "ModulatedRenewal", "SplineTransformer"]:
        for dataset in [
            "mimic",
            "mooc",
            "retweet",
            "simulated",
            "stackoverflow",
            "yelp",
        ]:
            subset = (
                runs[
                    (runs.model_name == model_name)
                    & (runs.dataset == dataset)
                    & (runs.seed == 42)
                ]
                .sort_values("best_qqdev_val")
                .reset_index(drop=True)
            )
            best_qqdev = subset.best_qqdev_val.min()
            best = subset[subset.best_qqdev_val == best_qqdev]
            if best.shape[0] > 1:
                best = best.sort_values(["n_knots", "block_size", "n_blocks", "n_embd"])
                best = best.iloc[[0]]
            best_runs.append(best)

    best_runs = pd.concat(best_runs, ignore_index=True)

    return best_runs


def multiple_best_experiment(
    best_runs, runs, min_memory=16, min_memory_transformer=30, gpus=None
):
    for i, best in best_runs.iterrows():
        learning_rate = best["learning_rate"]
        n_knots = best["n_knots"]
        n_embd = best["n_embd"]
        block_size = best["block_size"]
        n_blocks = best["n_blocks"]
        model_name = best["model_name"]
        dataset = best["dataset"]
        if model_name == "SplineTransformer":
            _min_memory = min_memory_transformer
        else:
            _min_memory = min_memory
        if (model_name == "SplineTransformer") and (dataset in ["mooc", "yelp"]):
            b = 10
            vb = 10
        else:
            b = 50
            vb = 50
        for seed in [42, 32, 54, 93, 91]:
            if is_config_trained(
                runs,
                learning_rate=learning_rate,
                n_knots=n_knots,
                n_embd=n_embd,
                block_size=block_size,
                n_blocks=n_blocks,
                model_name=model_name,
                dataset=dataset,
                seed=seed,
            ):
                continue
            print(
                f"Training {dataset}, {model_name}, nb{n_blocks}, bs{block_size}, nk{n_knots}, ne{n_embd}, s{seed}"
            )
            free_gpus = check_gpu_usage(min_memory=_min_memory, gpus=gpus)
            while len(free_gpus) == 0:
                print("No GPUs available, sleeping.")
                time.sleep(120)
                free_gpus = check_gpu_usage(min_memory=_min_memory, gpus=gpus)
            gpu = free_gpus[0]
            expname = f"/tmp/exp{gpu}-train-{dataset}-{model_name}-nb{n_blocks}-bs{block_size}-nk{n_knots}-ne{n_embd}-s{seed}"
            print(f"Launching {expname}.")
            subprocess.run(
                [
                    "dtach",
                    "-n",
                    expname,
                    "python",
                    "./bin/experiment.py",
                    dataset,
                    model_name,
                    "-d",
                    f"{gpu:d}",
                    "-s",
                    f"{seed:d}",
                    "-lr",
                    f"{learning_rate:E}",
                    "-nb",
                    f"{n_blocks}",
                    "-bs",
                    f"{block_size}",
                    "-nk",
                    f"{n_knots}",
                    "-ne",
                    f"{n_embd}",
                    "-e",
                    "1_000",
                    "-b",
                    f"{b}",
                    "-vb",
                    f"{vb}",
                    "-o",
                    "checkpoints",
                ]
            )
            time.sleep(10)


def multiple_evaluate(
    best_runs, runs, min_memory=16, min_memory_transformer=30, gpus=None
):
    """Evaluate multiple experiments."""
    for i, best in best_runs.iterrows():
        learning_rate = best["learning_rate"]
        n_knots = best["n_knots"]
        n_embd = best["n_embd"]
        block_size = best["block_size"]
        n_blocks = best["n_blocks"]
        model_name = best["model_name"]
        dataset = best["dataset"]
        if model_name == "SplineTransformer":
            _min_memory = min_memory_transformer
        else:
            _min_memory = min_memory
        if (model_name == "SplineTransformer") and (dataset in ["mooc", "yelp"]):
            vb = 5
        else:
            vb = 5
        subset = filter_runs(
            runs,
            learning_rate=learning_rate,
            n_knots=n_knots,
            n_embd=n_embd,
            block_size=block_size,
            n_blocks=n_blocks,
            model_name=model_name,
            dataset=dataset,
        )
        subset = subset.sort_values("train_time", ascending=False)
        subset = subset.drop_duplicates(
            subset=[
                "learning_rate",
                "n_knots",
                "n_embd",
                "block_size",
                "n_blocks",
                "model_name",
                "dataset",
                "seed",
            ]
        )
        # assert subset.shape[0] == 5
        for _, row in subset.iterrows():
            ckp = row["ckp"]
            log_path = ckp.parent / f"{ckp.stem}-evaluation.log"
            if log_path.exists():
                continue
            ckp_id = re.search(r"\w+-2023\d{4}-\d{6}", str(ckp))
            ckp_id = ckp_id.group(0)
            free_gpus = check_gpu_usage(gpus=gpus, min_memory=_min_memory)
            while len(free_gpus) == 0:
                print("No GPUs available, sleeping.")
                time.sleep(120)
                free_gpus = check_gpu_usage(gpus=gpus, min_memory=_min_memory)
            gpu = free_gpus[0]
            expname = f"/tmp/exp{gpu}-eval-{ckp_id}"
            print(f"Launching {expname}.")
            subprocess.run(
                [
                    "dtach",
                    "-n",
                    expname,
                    "python",
                    "./bin/evaluate.py",
                    f"{ckp}",
                    "-d",
                    f"{gpu:d}",
                    "-vb",
                    f"{vb}",
                ]
            )
            time.sleep(20)


def multiple_benchmark(runs, min_memory=16, min_memory_transformer=30, gpus=None):
    """Benchmark event generate time."""
    for i, best in best_runs.iterrows():
        learning_rate = best["learning_rate"]
        n_knots = best["n_knots"]
        n_embd = best["n_embd"]
        block_size = best["block_size"]
        n_blocks = best["n_blocks"]
        model_name = best["model_name"]
        dataset = best["dataset"]
        if model_name == "SplineTransformer":
            _min_memory = min_memory_transformer
        else:
            _min_memory = min_memory
        if (model_name == "SplineTransformer") and (dataset in ["mooc", "yelp"]):
            vb = 10
        else:
            vb = 10
        subset = filter_runs(
            runs,
            learning_rate=learning_rate,
            n_knots=n_knots,
            n_embd=n_embd,
            block_size=block_size,
            n_blocks=n_blocks,
            model_name=model_name,
            dataset=dataset,
        )
        subset = subset.sort_values("train_time", ascending=False)
        subset = subset.drop_duplicates(
            subset=[
                "learning_rate",
                "n_knots",
                "n_embd",
                "block_size",
                "n_blocks",
                "model_name",
                "dataset",
                "seed",
            ]
        )
        # assert subset.shape[0] == 5
        for _, row in subset.iterrows():
            ckp = row["ckp"]
            log_path = ckp.parent / f"{ckp.stem}-benchmark.log"
            if log_path.exists():
                continue
            ckp_id = re.search(r"\w+-2023\d{4}-\d{6}", str(ckp))
            ckp_id = ckp_id.group(0)
            free_gpus = check_gpu_usage(gpus=gpus, min_memory=_min_memory)
            while len(free_gpus) == 0:
                print("No GPUs available, sleeping.")
                time.sleep(120)
                free_gpus = check_gpu_usage(gpus=gpus, min_memory=_min_memory)
            gpu = free_gpus[0]
            expname = f"/tmp/exp{gpu}-bench-{model_name}-{ckp_id}"
            print(f"Launching {expname}.")
            subprocess.run(
                [
                    "dtach",
                    "-n",
                    expname,
                    "python",
                    "./bin/benchmark.py",
                    f"{ckp}",
                    "-d",
                    f"{gpu:d}",
                    "-vb",
                    f"{vb}",
                ]
            )
            time.sleep(20)


if __name__ == "__main__":
    runs = get_all_runs()
    multiple_experiment(runs)
    best_runs = select_best_model(runs)
    best_runs.to_pickle("./checkpoints/best-runs.pkl")
    multiple_best_experiment(best_runs, runs)
    multiple_evaluate(best_runs, runs, gpus=[0, 1, 2, 3, 4, 6, 7])
