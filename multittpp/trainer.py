"""
Implementation inspired by GNTPP.
https://github.com/BIRD-TAO/GNTPP/blob/main/trainers/trainer.py
"""
import os
import logging
import tempfile
import numpy as np

import torch
import torch.utils.data as data_utils
import torch.nn.functional as F

from time import time
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from typing import Optional
from functools import partial
from prettytable import PrettyTable

from . import data as multittpp_data
from . import models
from .utils import get_logger
from .config import Config


class Trainer:
    def __init__(
        self,
        data: dict[str, data_utils.DataLoader],
        model: models.TransformedExponential,
        config: Config,
        checkpoint_dict: Optional[dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        log: bool = True,
    ):
        self.data = data
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = (
            torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            if optimizer is None
            else optimizer
        )

        self.lr_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.5, patience=100, verbose=True
            )
            if lr_scheduler is None
            else lr_scheduler
        )

        self.checkpoint_dict = {
            "model_state_dict": None,
            "config": vars(self.config),
            "best_train_loss": float("inf"),
            "best_val_loss": float("inf"),
            "best_test_loss": float("inf"),
            "best_grad_norm_raw": float("inf"),
            "best_grad_norm_clip": float("inf"),
            "train_losses": [],
            "val_losses": [],
            "grad_norms_raw": [],
            "grad_norms_clip": [],
            "epochs": 0,
            "duration": 0,
            "iter_per_second": None,
        }

        if checkpoint_dict is not None:
            for k in self.checkpoint_dict:
                if k not in checkpoint_dict:
                    raise ValueError(f"Key {k} missing from checkpoint_dict.")
            self.checkpoint_dict = deepcopy(checkpoint_dict)

        self.logger = get_logger(__name__, config.log, level=logging.INFO)
        if not log:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

        self._init_evaluation_func()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint: str | Path,
        device: int = Config.device,
        seed: int = None,
        val_batch_size: int = None,
        log_path: Optional[str | Path] = None,
        log: bool = True,
    ):
        if isinstance(checkpoint, str):
            checkpoint = Path(checkpoint)

        assert (
            checkpoint.exists and checkpoint.is_file()
        ), f"The checkpoint {checkpoint} does not exist or is not a file."

        if isinstance(device, int) or isinstance(device, str):
            device = torch.device(device)
        checkpoint_dict = torch.load(checkpoint, map_location=device)

        config = Config(**checkpoint_dict["config"])
        config.device = device
        if seed is not None:
            config.seed = seed
        if val_batch_size is not None:
            config.val_batch_size = val_batch_size
        config.output = None
        config.checkpoint = None
        config.log = log_path
        checkpoint_dict["config"] = vars(config)

        if config.model_name is None:
            raise ValueError(f"Loaded config is missing model_name")

        model = getattr(models, config.model_name)(**vars(config))
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        model.to(device)

        if config.dataset is None:
            raise ValueError(f"Loaded config is missing dataset")

        (data, _, _, _, _) = multittpp_data.load_dataset(
            dataset=config.dataset,
            fixed_normalization=config.fixed_normalization,
            N_min=config.block_size,
            batch_size=config.batch_size,
            val_batch_size=config.val_batch_size,
            device=config.device,
        )

        trainer = cls(
            data=data,
            model=model,
            config=config,
            checkpoint_dict=checkpoint_dict,
            log=log,
        )

        trainer.logger.info(f"{config}")

        return trainer

    def _init_evaluation_func(self):
        self.eval_dict = {
            "LOSS": self.compute_loss,
            # NLL stands for negative log-likelihood
            "NLL": self.compute_nll,
            # CE stands for cross-entropy
            "CE_COND": partial(self.compute_ce, k_prob=None),
            "CE_SAMPLED_NONCOND": self.compute_ce,
            "CE_EXPECTED_NONCOND": self.compute_ce,
            "MAPE_SAMPLED": partial(self.compute_pred_time_error, kind="mape"),
            "RMSE_SAMPLED": partial(self.compute_pred_time_error, kind="mse"),
            "MAPE_EXPECTED": partial(self.compute_pred_time_error, kind="mape"),
            "RMSE_EXPECTED": partial(self.compute_pred_time_error, kind="mse"),
            "CRPS": self.compute_crps,
            # accuracy is the same as F1-micro
            # see: https://zephyrnet.com/micro-macro-weighted-averages-of-f1-score-clearly-explained/
            "TOP1_ACC_COND": partial(self.compute_k_acc, k_prob=None, top=1),
            "TOP3_ACC_COND": partial(self.compute_k_acc, k_prob=None, top=3),
            "TOP1_ACC_SAMPLED_NONCOND": partial(self.compute_k_acc, top=1),
            "TOP3_ACC_SAMPLED_NONCOND": partial(self.compute_k_acc, top=3),
            "TOP1_ACC_EXPECTED_NONCOND": partial(self.compute_k_acc, top=1),
            "TOP3_ACC_EXPECTED_NONCOND": partial(self.compute_k_acc, top=3),
            "QQDEV": self.compute_cumulative_risk,
        }

    def batch(
        self,
        dataset: str,
        grad: bool,
        verbose: bool = False,
        postfix: Optional[dict] = None,
        total: Optional[int | float] = None,
    ):
        device = self.config.device
        data_iterator = self.data["{}_loader".format(dataset)]

        if grad:
            with tqdm(
                data_iterator,
                total=total,
                dynamic_ncols=True,
                postfix=postfix,
                leave=False,
                unit="batch",
                disable=(not verbose),
            ) as t:
                with torch.device(device):
                    for i, batch in enumerate(t):
                        yield t, i, batch
        else:
            with torch.no_grad():
                with tqdm(
                    data_iterator,
                    total=total,
                    dynamic_ncols=True,
                    postfix=postfix,
                    leave=False,
                    unit="batch",
                    disable=(not verbose),
                ) as t:
                    with torch.device(device):
                        for i, batch in enumerate(t):
                            yield t, i, batch

    def save_model(
        self,
        checkpoint_path: str,
        epochs: int,
        training_duration: float,
        best_train_loss: float,
        best_val_loss: float,
        best_test_loss: float,
        best_grad_norm_raw: float,
        best_grad_norm_clip: float,
        train_losses: list[float],
        val_losses: list[float],
        grad_norms_raw: list[float],
        grad_norms_clip: list[float],
    ) -> None:
        if epochs is None or training_duration is None:
            iter_per_second = None
        else:
            iter_per_second = epochs / training_duration
        checkpoint_dict = {
            "model_state_dict": self.model.state_dict(),
            "config": vars(self.config),
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "best_test_loss": best_test_loss,
            "best_grad_norm_raw": best_grad_norm_raw,
            "best_grad_norm_clip": best_grad_norm_clip,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "grad_norms_raw": grad_norms_raw,
            "grad_norms_clip": grad_norms_clip,
            "epochs": epochs,
            "duration": training_duration,
            "iter_per_second": iter_per_second,
        }
        self.checkpoint_dict = checkpoint_dict
        torch.save(checkpoint_dict, checkpoint_path)

    def load_model(self, checkpoint_path: str) -> None:
        self.logger.info(f"Loading from checkpoint path '{checkpoint_path}'.")
        if torch.cuda.is_available():
            checkpoint_dict = torch.load(checkpoint_path, map_location="cuda")
        else:
            checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model = getattr(models, checkpoint_dict["config"]["model_name"])(
            **vars(self.config)
        )
        self.model.load_state_dict(checkpoint_dict["model_state_dict"])
        self.model.to(self.config.device)
        self.checkpoint_dict = checkpoint_dict

    def count_parameters(self) -> tuple[PrettyTable, int]:
        summary_table = PrettyTable(["Parameters", "#"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            name_parts = name.split(".")
            if name_parts[0] == "transforms":
                ix = int(name_parts[1])
                name_parts[
                    1
                ] = f"{name_parts[1]}[{type(self.model.transforms[ix]).__name__}]"
            name = ".".join(name_parts)
            params = parameter.numel()
            summary_table.add_row([name, f"{params}"])
            total_params += params
        return summary_table, total_params

    def train(self) -> None:
        wait = 0

        if self.config.checkpoint is None:
            _, checkpoint_path = tempfile.mkstemp(
                suffix="pth", prefix="multittpp-trainer"
            )
        else:
            checkpoint_path = self.config.checkpoint

        epoch_train_metrics = self.evaluate("train", ["LOSS"], verbose=False)
        epoch_train_loss = epoch_train_metrics["LOSS"]
        epoch_train_N = epoch_train_metrics["N"]
        epoch_train_loss_per_event = epoch_train_loss / epoch_train_N
        self.checkpoint_dict["train_losses"].append(epoch_train_loss_per_event)

        epoch_val_metrics = self.evaluate("val", ["LOSS"], verbose=False)
        epoch_val_loss = epoch_val_metrics["LOSS"]
        epoch_val_N = epoch_val_metrics["N"]
        epoch_val_loss_per_event = epoch_val_loss / epoch_val_N
        self.checkpoint_dict["val_losses"].append(epoch_val_loss_per_event)

        epoch_test_metrics = self.evaluate("test", ["LOSS"], verbose=False)
        epoch_test_loss = epoch_test_metrics["LOSS"]
        epoch_test_N = epoch_test_metrics["N"]
        epoch_test_loss_per_event = epoch_test_loss / epoch_test_N

        best_val_loss = self.checkpoint_dict["best_val_loss"]

        if epoch_val_loss_per_event < best_val_loss:
            self.checkpoint_dict["best_train_loss"] = epoch_train_loss_per_event
            self.checkpoint_dict["best_val_loss"] = epoch_val_loss_per_event
            self.checkpoint_dict["best_test_loss"] = epoch_test_loss_per_event

        wait = 0
        epoch_num = self.checkpoint_dict["epochs"]
        epoch_start = epoch_num
        best_epoch = epoch_num - 1

        postfix = {
            "best_ep": 0,
            "loss_train": 0.0,
            "loss_val": 0.0,
            "grad_norm": 0.0,
            "mem": 0.0,
            "wait": 0.0,
        }

        self.log_training(best_epoch, wait=wait, unit="ep")

        duration_start = self.checkpoint_dict["duration"]
        training_start = time()

        with tqdm(
            range(epoch_start, epoch_start + self.config.epochs),
            dynamic_ncols=True,
            postfix=postfix,
            leave=False,
            unit="ep",
        ) as t:
            for epoch_num in t:
                # optimization
                self.model.train()

                epoch_train_loss = 0
                epoch_train_N = 0

                batches_N = len(self.data["train_loader"])

                for _, i, batch in self.batch(dataset="train", grad=True, verbose=True):
                    epoch_train_N += torch.sum(batch.seq_lengths).detach().item()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = self.compute_loss(batch)
                    loss.backward()

                    if i == batches_N - 1:
                        batch_grad_norm_raw = [
                            p.grad.detach().flatten()
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                        batch_grad_norm_raw = (
                            torch.cat(batch_grad_norm_raw).norm().detach().item()
                        )
                        postfix["grad_norm"] = batch_grad_norm_raw

                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    if i == batches_N - 1:
                        batch_grad_norm_clip = [
                            p.grad.detach().flatten()
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                        batch_grad_norm_clip = (
                            torch.cat(batch_grad_norm_clip).norm().detach().item()
                        )

                    self.optimizer.step()
                    epoch_train_loss += loss.detach()

                # reduce the learning rate if training loss stops decreasing
                self.lr_scheduler.step(loss)

                # post-mortem
                self.model.eval()

                epoch_train_loss_per_event = (
                    epoch_train_loss.detach().item() / epoch_train_N
                )
                self.checkpoint_dict["train_losses"].append(epoch_train_loss_per_event)
                postfix["loss_train"] = epoch_train_loss_per_event

                # gradient norm
                self.checkpoint_dict["grad_norms_raw"].append(batch_grad_norm_raw)
                self.checkpoint_dict["grad_norms_clip"].append(batch_grad_norm_clip)
                postfix["grad_norm"] = batch_grad_norm_raw

                # validation
                epoch_val_metrics = self.evaluate("val", ["LOSS"], verbose=False)
                epoch_val_loss = epoch_val_metrics["LOSS"]
                epoch_val_N = epoch_val_metrics["N"]

                epoch_val_loss_per_event = epoch_val_loss / epoch_val_N
                self.checkpoint_dict["val_losses"].append(epoch_val_loss_per_event)
                postfix["loss_val"] = epoch_val_loss_per_event

                # best perfomance
                if epoch_val_loss_per_event < self.checkpoint_dict["best_val_loss"]:
                    wait = 0
                    postfix["best_ep"] = epoch_num
                    best_val_loss = epoch_val_loss_per_event
                    epoch_test_metrics = self.evaluate("test", ["LOSS"], verbose=False)
                    epoch_test_loss = epoch_test_metrics["LOSS"]
                    epoch_test_N = epoch_test_metrics["N"]
                    epoch_test_loss_per_event = epoch_test_loss / epoch_test_N
                    t.clear()
                    training_duration = (time() - training_start) + duration_start
                    self.save_model(
                        checkpoint_path=checkpoint_path,
                        epochs=epoch_num + 1,
                        training_duration=training_duration,
                        best_train_loss=epoch_train_loss_per_event,
                        best_val_loss=best_val_loss,
                        best_test_loss=epoch_test_loss_per_event,
                        best_grad_norm_raw=batch_grad_norm_raw,
                        best_grad_norm_clip=batch_grad_norm_clip,
                        train_losses=self.checkpoint_dict["train_losses"],
                        val_losses=self.checkpoint_dict["val_losses"],
                        grad_norms_raw=self.checkpoint_dict["grad_norms_raw"],
                        grad_norms_clip=self.checkpoint_dict["grad_norms_clip"],
                    )
                    self.log_training(epoch_num, wait=wait, unit="ep")
                    t.refresh()

                # early stopping
                else:
                    wait += 1
                    if np.isnan(epoch_val_loss_per_event):
                        self.logger.info(f"Validation loss is NaN.")
                    if wait == self.config.patience or np.isnan(
                        epoch_val_loss_per_event
                    ):
                        t.close()
                        self.logger.info(f"Early stopping at epoch: {epoch_num}")
                        self.checkpoint_dict["duration"] = (
                            time() - training_start
                        ) + duration_start
                        self.log_training(epoch_num, wait=wait, unit="ep")
                        self.load_model(checkpoint_path)
                        return

                m = torch.cuda.memory_allocated(self.config.device) / 1e6
                postfix["mem"] = m

                # logging
                postfix["wait"] = wait
                t.set_postfix(postfix)

                if (epoch_num % self.config.log_every) == self.config.log_every - 1:
                    t.clear()
                    self.checkpoint_dict["duration"] = (
                        time() - training_start
                    ) + duration_start
                    self.log_training(epoch_num, wait=wait, unit="ep")
                    t.refresh()

        self.checkpoint_dict["duration"] = (time() - training_start) + duration_start
        self.log_training(epoch_num, wait=wait, unit="ep")
        self.load_model(checkpoint_path)

    def log_training(self, epoch_num: int, wait: int, unit: str = "it") -> None:
        epochs = epoch_num + 1
        training_duration = self.checkpoint_dict["duration"]
        if training_duration == 0:
            iter_per_second = None
            iter_per_second_msg = f", ?{unit}/s"
        else:
            # training_duration = time() - training_start
            iter_per_second = epochs / training_duration
            if iter_per_second > 1:
                iter_per_second_msg = f", {iter_per_second:.2f}{unit}/s"
            else:
                iter_per_second_msg = f", {1/iter_per_second:.2f}s/{unit}"
        training_duration_msg = self.pretty_duration(training_duration)
        epochs_msg = f"%{len(str(epochs)):d}d" % epochs
        msg = (
            f"{epochs_msg} / {self.config.epochs} ["
            + f"{training_duration_msg}"
            + f"{iter_per_second_msg}"
            + f", best_ep={self.checkpoint_dict['epochs'] - 1:,d}"
            + f", best_loss_train={tqdm.format_num(self.checkpoint_dict['best_train_loss'])}"
            + f", best_loss_val={tqdm.format_num(self.checkpoint_dict['best_val_loss'])}"
            + f", best_loss_test={tqdm.format_num(self.checkpoint_dict['best_test_loss'])}"
            + f", best_grad_norm_raw={tqdm.format_num(self.checkpoint_dict['best_grad_norm_raw'])}"
            + f", wait={tqdm.format_num(wait)}"
            + f"]"
        )
        self.logger.info(msg)

    def pretty_duration(self, delta: float) -> str:
        hours, remainder = divmod(delta, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    def evaluate(
        self,
        dataset: str,
        metrics: list[str] = ["LOSS"],
        msg_prefix: str = "",
        verbose: bool = False,
    ) -> dict[str, float]:
        self.model.eval()

        report_metric = {}

        need_pred_time = ["MAPE_SAMPLED", "RMSE_SAMPLED"]
        compute_pred_time = False
        need_expected_time = ["MAPE_EXPECTED", "RMSE_EXPECTED"]
        compute_expected_time = False
        need_event_prob_sampled = [
            "CE_SAMPLED_NONCOND",
            "TOP1_ACC_SAMPLED_NONCOND",
            "TOP3_ACC_SAMPLED_NONCOND",
        ]
        compute_event_prob_sampled = False
        need_event_prob_expected = [
            "CE_EXPECTED_NONCOND",
            "TOP1_ACC_EXPECTED_NONCOND",
            "TOP3_ACC_EXPECTED_NONCOND",
        ]
        compute_event_prob_expected = False

        if verbose:
            self.logger.info(f"Evaluation started: {dataset} dataset")

        for metric in metrics:
            if metric in need_pred_time:
                compute_pred_time = True
            if metric in need_expected_time:
                compute_expected_time = True
            if metric in need_event_prob_sampled:
                compute_pred_time = True
                compute_event_prob_sampled = True
            if metric in need_event_prob_expected:
                compute_expected_time = True
                compute_event_prob_expected = True
            report_metric[metric] = 0 if metric != "QQDEV" else []

        N = 0

        postfix = {
            "mem": 0.0,
        }

        for t, i, batch in self.batch(
            dataset=dataset, grad=False, verbose=True, postfix=postfix
        ):
            N += torch.sum(batch.seq_lengths).detach().item()

            if compute_pred_time:
                pred_time, event_prob_sampled = self.predict_event(
                    batch, time=compute_pred_time, prob=compute_event_prob_sampled
                )

            if compute_expected_time:
                expected_time, event_prob_expected = self.expected_event(
                    batch, time=compute_expected_time, prob=compute_event_prob_expected
                )

            for metric in metrics:
                if metric in need_pred_time:
                    value = self.eval_dict[metric](batch, pred_time)
                elif metric in need_expected_time:
                    value = self.eval_dict[metric](batch, expected_time)
                elif metric in need_event_prob_sampled:
                    value = self.eval_dict[metric](batch, k_prob=event_prob_sampled)
                elif metric in need_event_prob_expected:
                    value = self.eval_dict[metric](batch, k_prob=event_prob_expected)
                else:
                    value = self.eval_dict[metric](batch)

                if metric != "QQDEV":
                    report_metric[metric] += value.detach().item()
                else:
                    report_metric[metric].append(value)

                del value

                with torch.cuda.device("cuda:{}".format(self.config.device)):
                    torch.cuda.empty_cache()

            self.optimizer.zero_grad()

            m = torch.cuda.memory_allocated(self.config.device) / 1e6
            postfix["mem"] = m
            t.set_postfix(postfix)

        for m in ["RMSE_SAMPLED", "RMSE_EXPECTED"]:
            if m in metrics:
                report_metric[m] = np.sqrt(report_metric[m])

        if "QQDEV" in metrics:
            cumrisk_samples = torch.cat(report_metric["QQDEV"]).sort().values
            probs = torch.linspace(0, 1, 101)[1:-1].to(cumrisk_samples)
            estimate_quantiles = torch.quantile(cumrisk_samples, probs)
            exp1_quantiles = -(1 - probs).log()
            report_metric["QQDEV"] = (
                (estimate_quantiles - exp1_quantiles).abs().mean().detach().item()
            )

        N = int(N)
        report_metric["N"] = N

        if verbose:
            msg = msg_prefix
            msg += f"\n\tNumber of events: {N:,d}"
            for m in metrics:
                if m == "QQDEV":
                    msg += f"\n\t{m}: {report_metric[m]}"
                elif "RMSE" in m:
                    msg += f"\n\t{m}: {report_metric[m] / np.sqrt(N):.5f}"
                else:
                    msg += f"\n\t{m}: {report_metric[m] / N:.5f}"

            self.logger.info(msg)

        return report_metric

    def predict_event(self, batch, time, prob):
        y = batch.in_times
        k = batch.in_types
        n_samples = self.config.n_samples
        y_sample, k_sample = self.model.sample(y, k, n_samples)
        if time or prob:
            y_pred = self.model.predict_event_time(y_sample)
        else:
            y_pred = None
        if prob:
            k_prob = self.model.predict_event_prob(k_sample)
        else:
            k_prob = None
        # we ignore cold start prediction (ie prediction from time 0)
        # this is to align with GNTPP computation
        return y_pred[:, 1:], k_prob[:, 1:]

    def expected_event(self, batch, time, prob):
        y = batch.in_times
        k = batch.in_types
        n_samples = self.config.n_samples
        t_max = self.config.t_max_normalization
        if time:
            y_expected, k_expected = self.model.expected_event(y, k, t_max, n_samples)
        else:
            y_expected = None
        if prob:
            k_prob = self.model.expected_event_prob(y, k, t_max)
        else:
            k_prob = None
        # we ignore cold start prediction (ie prediction from time 0)
        # this is to align with GNTPP computation
        return y_expected[:, 1:], k_prob[:, 1:]

    def compute_loss(self, batch):
        return self.compute_nll(batch)

    def compute_nll(self, batch):
        y = batch.in_times
        k = batch.in_types
        yT = batch.last_times
        return self.model.loss(y, k, yT)

    def _normalize_eval_data(self, input):
        return input * (
            self.config.fixed_normalization / self.config.t_max_normalization
        )

    def compute_ce(self, batch, k_prob=None):
        if k_prob is None:
            raise NotImplementedError
        k = batch.out_types
        return self._compute_ce(k, k_prob)

    def _compute_ce(self, k, k_prob):
        try:
            # assign an arbitrary value for non-events to ensure Categorical does not raise an error
            k_prob[k == self.model.n_marks] = 1 / self.model.n_marks
            k_dist = torch.distributions.Categorical(probs=k_prob)
            # assign an arbitrary mark for non-events to ensure log_prob does not raise an error
            _k = k.clone()
            _k[k == self.model.n_marks] = 0
            ce = -(k_dist.log_prob(_k) * (k < self.model.n_marks)).sum()
        except:
            return torch.tensor(-1)
        return ce

    def compute_pred_time_error(self, batch, y_pred, kind):
        y_dts = self._normalize_eval_data(batch.out_dts)
        y_dts_pred = self._normalize_eval_data(y_pred) - F.pad(
            self._normalize_eval_data(batch.in_times), (1, -1)
        )
        k = batch.out_types
        return self._pred_time_error(y_dts, k, y_dts_pred, kind)

    def _pred_time_error(self, y_dts, k, y_dts_pred, kind):
        # relative absolute error
        try:
            self.logger
            errors = torch.divide(
                (y_dts_pred.clamp(min=0, max=self.config.fixed_normalization) - y_dts),
                y_dts + 1e-7,
            )
            if kind == "mape":
                errors = (errors.abs() * (k < self.model.n_marks)).clamp(
                    max=2 * self.config.fixed_normalization
                )
            elif kind == "mse":
                errors = ((errors**2) * (k < self.model.n_marks)).clamp(
                    max=(2 * self.config.fixed_normalization) ** 2
                )
            else:
                raise ValueError(f"Unknown error kind: {kind}")
            return errors.sum()
        except:
            return torch.tensor(-1)

    def compute_crps(self, batch):
        try:
            y_dts = self._normalize_eval_data(batch.out_dts)
            k = batch.out_types

            y_sample, _ = self.model.sample(
                batch.in_times, batch.in_types, self.config.n_samples
            )
            y_sample = self._normalize_eval_data(y_sample)
            y_dts_sample = y_sample - F.pad(
                self._normalize_eval_data(batch.in_times), (1, -1)
            )
            y_dts_sample[y_dts_sample >= self.config.fixed_normalization] = (
                2 * self.config.fixed_normalization
            )

            k_mask = k < self.model.n_marks

            diff_sample = ((y_dts_sample - y_dts) * k_mask).abs().sum(
                dim=0
            ) / self.config.n_samples
            diff_sample = diff_sample.sum()

            diff_distri = (
                (
                    (y_dts_sample * k_mask)[None, ...]
                    - (y_dts_sample * k_mask)[:, None, ...]
                )
            ).abs().sum(dim=(0, 1)) / (self.config.n_samples**2 * 2)
            diff_distri = diff_distri.sum()

            return diff_sample - diff_distri
        except:
            return torch.tensor(-1)

    def compute_k_acc(self, batch, k_prob=None, top=1):
        if k_prob is None:
            raise NotImplementedError
        k = batch.out_types
        return self._top_k_acc(k, k_prob, top)

    def _top_k_acc(
        self,
        k,
        k_prob,
        top,
    ):
        try:
            k_pred = self.model.predict_event_type(k_prob, top)
            correct = k_pred.eq(k.unsqueeze(-1).expand_as(k_pred)) * (
                k < self.model.n_marks
            ).unsqueeze(-1)
            correct_k = correct.view(-1).float().sum(0)
            return correct_k
        except:
            return torch.tensor(-1)

    def compute_cumulative_risk(self, batch):
        y = batch.in_times
        k = batch.in_types
        cumulative_risk = self.model.cumulative_risk_func(y, k)
        cumulative_risk = cumulative_risk[k < self.model.n_marks].flatten()
        return cumulative_risk

    def empirical_quantiles(self, dataset: str, n_threshold: int = 0):
        x = [[] for _ in range(self.config.n_marks)]
        flat_x = []

        for _, _, batch in self.batch(dataset, grad=False, verbose=True):
            y = batch.in_times
            k = batch.in_types
            _x = self.model.inverse(y, k)
            flat_x.append(_x[k < self.config.n_marks].flatten())
            for mark in range(self.config.n_marks):
                x[mark].append(_x[mark == k].flatten())

        probs = torch.linspace(0, 1, 101)[1:-1].to(y.device)

        flat_quant_pred = torch.quantile(torch.cat(flat_x), probs)

        quant_pred = [[] for _ in range(self.config.n_marks)]

        for mark in range(self.config.n_marks):
            x[mark] = torch.cat(x[mark])
            if (len(x[mark]) != 0) and (x[mark].shape[0] > n_threshold):
                x[mark] = x[mark].sort().values
                quant_pred[mark] = torch.quantile(x[mark], probs)
            else:
                quant_pred[mark] = torch.tensor([], dtype=x[mark].dtype)

        return probs, quant_pred, flat_quant_pred

    def generate(self, batch, start_ix, n_samples, N_min):
        self.model.eval()
        y = batch.in_times
        k = batch.in_types
        k[:, start_ix:] = self.model.n_marks
        y_gen, k_gen = self.model.generate(y=y, k=k, n_samples=n_samples, N_min=N_min)
        return y_gen, k_gen
