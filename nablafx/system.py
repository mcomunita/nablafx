import os
import torch
import random
import auraloss

# import pytorch_lightning as pl
import lightning as pl
import torchmetrics as tm
import wandb

from typing import List, Optional
from nablafx.plotting import plot_gb_model, plot_frequency_response_steps
from nablafx.models import BlackBoxModel, GreyBoxModel
from nablafx.loss import TimeAndFrequencyDomainLoss, WeightedMultiLoss
from frechet_audio_distance import FrechetAudioDistance

import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)


# -----------------------------------------------------------------------------
# Base System
# -----------------------------------------------------------------------------


class BaseSystem(pl.LightningModule):
    def __init__(
        self,
        loss: torch.nn.Module,
        lr: float = 1e-4,
        log_media_every_n_steps: int = 10000,
        use_callbacks: bool = False,
    ):
        """
        Base system for audio effect modeling.

        Args:
            loss: Loss function to use
            lr: Learning rate
            log_media_every_n_steps: Steps between media logging (ignored if use_callbacks=True)
            use_callbacks: If True, relies on callbacks for logging instead of built-in methods.
                          When True, the following methods become no-ops:
                          - compute_and_log_metrics
                          - log_audio
                          - log_frequency_response
                          - compute_and_log_fad
        """
        super().__init__()
        self.loss = loss
        self.lr = lr
        self.log_media_every_n_steps = log_media_every_n_steps
        self.log_media_counter = 0
        self.log_input_and_target_flag = True
        self.use_callbacks = use_callbacks

        # metrics (only used if not using callbacks)
        if not self.use_callbacks:
            self.metrics = {
                "mae": tm.MeanAbsoluteError(),
                "mape": tm.MeanAbsolutePercentageError(),
                "mse": tm.MeanSquaredError(),
                "cossim": tm.CosineSimilarity(),
                "logcosh": auraloss.time.LogCoshLoss(),
                "esr": auraloss.time.ESRLoss(),
                "dcloss": auraloss.time.DCLoss(),
            }
        else:
            self.metrics = {}
            print("📋 Using callback-based logging - built-in logging methods disabled")

    def forward(self, input: torch.Tensor, params: torch.Tensor, train: bool = False):
        return self.model(input, params, train=train)

    def common_step(self, batch, batch_idx, mode="train"):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_train_start(self):
        # log gradients (always enabled)
        wandb.watch(self.model, log_freq=100)
        # Frequency response logging handled by callbacks if use_callbacks=True
        if not self.use_callbacks:
            pass  # self.log_frequency_response() # atm needs too much memory

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer (always enabled)
        # If using mixed precision, the gradients are already unscaled here
        norms = pl.pytorch.utilities.grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_end(self):
        # FAD computation handled by callbacks if use_callbacks=True
        if not self.use_callbacks:
            # self.log_frequency_response() # atm needs too much memory
            self.compute_and_log_fad(mode="val")

    def on_test_epoch_end(self):
        # Logging handled by callbacks if use_callbacks=True
        if not self.use_callbacks:
            self.log_frequency_response()
            self.compute_and_log_fad(mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)

        return [optimizer], [{"scheduler": lr_scheduler, "monitor": "loss/val/tot", "interval": "epoch", "frequency": 1}]

    def compute_and_log_loss(self, pred, target, mode, should_log=True):

        losses = self.loss(pred, target)

        # Handle different loss function formats
        if isinstance(self.loss, WeightedMultiLoss):
            # New WeightedMultiLoss format
            if isinstance(losses, tuple):
                # Multiple losses: (loss1, loss2, ..., total_loss)
                individual_losses = losses[:-1]
                tot_loss = losses[-1]
            else:
                # Single loss
                individual_losses = [losses]
                tot_loss = losses

            # Compute scaled losses for logging (unweighted values)
            scaled_losses = []
            loss_names = self.loss.get_loss_names()
            weights = self.loss.get_weights()

            for loss_val, weight in zip(individual_losses, weights):
                if weight > 0:
                    scaled_losses.append(loss_val / weight)
                else:
                    scaled_losses.append(loss_val)

            tot_loss_scaled = sum(scaled_losses)

        elif isinstance(self.loss, TimeAndFrequencyDomainLoss):
            # Original TimeAndFrequencyDomainLoss format (backward compatibility)
            td_loss, fd_loss = losses[0], losses[1]
            tot_loss = sum(losses)
            individual_losses = [td_loss, fd_loss]
            loss_names = ["l1", "mrstft"]

            td_loss_scaled = 0.0
            fd_loss_scaled = 0.0
            if self.loss.time_domain_weight > 0:
                td_loss_scaled = losses[0] / self.loss.time_domain_weight
            if self.loss.frequency_domain_weight > 0:
                fd_loss_scaled = losses[1] / self.loss.frequency_domain_weight
            tot_loss_scaled = td_loss_scaled + fd_loss_scaled
            scaled_losses = [td_loss_scaled, fd_loss_scaled]

        else:
            # Simple loss function (single value)
            if isinstance(losses, (tuple, list)):
                tot_loss = sum(losses)
                individual_losses = list(losses)
            else:
                tot_loss = losses
                individual_losses = [losses]
            loss_names = ["loss"]
            scaled_losses = individual_losses
            tot_loss_scaled = tot_loss

        if should_log:
            # Log total loss
            self.log(
                f"loss/{mode}/tot",
                tot_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            # Log individual losses
            for loss_val, loss_name in zip(individual_losses, loss_names):
                self.log(
                    f"loss/{mode}/{loss_name}",
                    loss_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )

            # Log scaled losses (unweighted)
            self.log(
                f"loss_scaled/{mode}/tot",
                tot_loss_scaled,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            for scaled_loss, loss_name in zip(scaled_losses, loss_names):
                self.log(
                    f"loss_scaled/{mode}/{loss_name}",
                    scaled_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )

        if torch.isnan(tot_loss):
            print("NaN loss encountered. Exiting...")
            sys.exit()

        return tot_loss

    def compute_and_log_metrics(self, pred, target, mode):
        """Compute and log metrics. No-op if using callbacks."""
        if self.use_callbacks:
            return  # Metrics handled by MetricsLoggingCallback

        for name, metric in self.metrics.items():
            metric = metric.to(self.device)
            if pred.dim() == 3:
                pred = pred.squeeze(1)
                target = target.squeeze(1)
            metric_value = metric(pred, target)

            metric_value = metric_value.detach().cpu()
            pred = pred.detach().cpu()
            target = target.detach().cpu()

            self.log(
                f"metrics/{mode}/{name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            if hasattr(metric, "reset"):
                metric.reset()
        pred, target, metric_value = None, None, None

    def log_audio(self, batch_idx, input, target, pred, mode):
        """Log audio samples. No-op if using callbacks."""
        if self.use_callbacks:
            return  # Audio logging handled by AudioLoggingCallback

        input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

        if mode == "test":  # log all audio in test mode
            if batch_idx < 10:  # log only first 10 batches
                print(f"\nLogging audio for {mode} batch {batch_idx}...")
                for i, _ in enumerate(input):
                    self.logger.experiment.log(
                        {
                            f"audio/{mode}/input/b{batch_idx}-{i}": wandb.Audio(
                                input[i].cpu().numpy()[0, :], 48000, caption=f"input_b{batch_idx}-{i}"
                            ),
                            f"audio/{mode}/target/b{batch_idx}-{i}": wandb.Audio(
                                target[i].cpu().numpy()[0, :], 48000, caption=f"target_b{batch_idx}-{i}"
                            ),
                            f"audio/{mode}/pred/b{batch_idx}-{i}": wandb.Audio(
                                pred[i].cpu().numpy()[0, :], 48000, caption=f"pred_b{batch_idx}-{i}"
                            ),
                        },
                        step=self.trainer.global_step,
                    )
        elif self.log_input_and_target_flag:  # log input and target only once
            for i, _ in enumerate(input):
                self.logger.experiment.log(
                    {
                        f"audio/{mode}/input/b{batch_idx}-{i}": wandb.Audio(
                            input[i].cpu().numpy()[0, :], 48000, caption=f"input_b{batch_idx}-{i}"
                        ),
                        f"audio/{mode}/target/b{batch_idx}-{i}": wandb.Audio(
                            target[i].cpu().numpy()[0, :], 48000, caption=f"target_b{batch_idx}-{i}"
                        ),
                        f"audio/{mode}/pred/b{batch_idx}-{i}": wandb.Audio(
                            pred[i].cpu().numpy()[0, :], 48000, caption=f"pred_b{batch_idx}-{i}"
                        ),
                    },
                    step=self.trainer.global_step,
                )
            self.log_input_and_target_flag = False
        else:
            for i, _ in enumerate(input):
                self.logger.experiment.log(
                    {
                        f"audio/{mode}/pred/b{batch_idx}-{i}": wandb.Audio(
                            pred[i].cpu().numpy()[0, :], 48000, caption=f"pred_b{batch_idx}-{i}"
                        ),
                    },
                    step=self.trainer.global_step,
                )
        input, target, pred = None, None, None

    def log_frequency_response(self):
        """Log frequency response plot. No-op if using callbacks."""
        if self.use_callbacks:
            return  # Frequency response handled by FrequencyResponseCallback

        print("\nLogging frequency response...")
        self.model.reset_states()
        with torch.no_grad():
            plot = plot_frequency_response_steps(self.model)
        self.logger.experiment.log({f"response/freq+phase": [wandb.Image(plot, caption=f"")]}, step=self.trainer.global_step)

    def compute_and_log_fad(self, mode):
        """Compute and log FAD scores. No-op if using callbacks."""
        if self.use_callbacks:
            return  # FAD computation handled by FADComputationCallback

        print("\nComputing and logging FAD...")
        run_dir = self.logger.experiment.dir
        pred_dir = os.path.join(run_dir, f"media/audio/audio/{mode}/pred")
        target_dir = os.path.join(run_dir, f"media/audio/audio/{mode}/target")

        parent_dir = os.path.abspath(os.getcwd())
        ckpt_dir = os.path.join(parent_dir, "checkpoints_fad")

        print(f"\n\nComputing FAD for {run_dir}...")

        fad_vggish = FrechetAudioDistance(
            os.path.join(ckpt_dir, "vggish"),
            model_name="vggish",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )

        fad_score_vggish = fad_vggish.score(
            target_dir,
            pred_dir,
        )

        fad_pann = FrechetAudioDistance(
            os.path.join(ckpt_dir, "pann"),
            model_name="pann",
            sample_rate=32000,
            verbose=False,
        )

        fad_score_pann = fad_pann.score(
            target_dir,
            pred_dir,
        )

        fad_clap = FrechetAudioDistance(
            os.path.join(ckpt_dir, "clap"),
            model_name="clap",
            submodel_name="630k-audioset",
            sample_rate=48000,
            verbose=False,
            enable_fusion=False,
        )

        fad_score_clap = fad_clap.score(
            target_dir,
            pred_dir,
        )

        fad_afxrep = FrechetAudioDistance(
            os.path.join(ckpt_dir, "afx-rep"),
            model_name="afx-rep",
            sample_rate=48000,
            verbose=False,
        )

        fad_score_afxrep = fad_afxrep.score(
            target_dir,
            pred_dir,
        )

        print(f"\nFAD score (vggish): {fad_score_vggish}")
        print(f"FAD score (pann): {fad_score_pann}")
        print(f"FAD score (clap): {fad_score_clap}")
        print(f"FAD score (afx-rep): {fad_score_afxrep}")

        self.logger.experiment.log(
            {
                f"metrics/{mode}/fad-vggish": fad_score_vggish,
                f"metrics/{mode}/fad-pann": fad_score_pann,
                f"metrics/{mode}/fad-clap": fad_score_clap,
                f"metrics/{mode}/fad-afxrep": fad_score_afxrep,
            },
            step=self.trainer.global_step,
        )


# -----------------------------------------------------------------------------
# Black Box System with truncated back-propagation through time
# at the start of each batch
# -----------------------------------------------------------------------------


class BlackBoxSystem(BaseSystem):
    def __init__(
        self,
        model: BlackBoxModel,
        loss: torch.nn.Module,
        lr: float = 1e-4,
        log_media_every_n_steps: int = 10000,
        use_callbacks: bool = False,
    ):
        super().__init__(loss, lr, log_media_every_n_steps, use_callbacks)
        self.model = model

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        mode: str,
    ):
        """Model step used for validation and training.
        Args:
            batch (Tuple[Tensor, Tensor, Tensor]): Batch items containing input audio (x) target audio (y).
            batch_idx (int): Index of the batch within the current epoch.
            mode (str): One of "train", "val", "test".
        """
        train = True if mode == "train" else False

        # reset hidden states for recurrent layers
        self.model.reset_states()
        if mode == "train":
            self.model.detach_states()

        # get batch
        if self.model.num_controls > 0:
            input, target, controls = batch
        else:
            input, target = batch
            controls = None

        # run the model
        pred = self(input, controls, train=train)

        # calculate loss
        tot_batch_loss = self.compute_and_log_loss(pred, target, mode)

        # calculate metrics
        self.compute_and_log_metrics(pred, target, mode)

        return tot_batch_loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="val")
        # log media (skip if using callbacks)
        if not self.use_callbacks and batch_idx == 0:
            if (self.trainer.global_step / self.log_media_every_n_steps) > self.log_media_counter:
                if self.model.num_controls > 0:
                    input, target, controls = batch
                else:
                    input, target = batch
                    controls = None

                self.model.reset_states()
                self.model.detach_states()
                pred = self(input, controls)

                input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

                self.log_audio(batch_idx, input, target, pred, "val")
                self.log_media_counter += 1

                input, target, pred = None, None, None
                if controls is not None:
                    controls = None
                # torch.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="test")

        # log media (skip if using callbacks)
        if not self.use_callbacks:
            if self.model.num_controls > 0:
                input, target, controls = batch
            else:
                input, target = batch
                controls = None

            self.model.reset_states()
            with torch.no_grad():
                pred = self(input, controls)
            input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

            self.log_audio(batch_idx, input, target, pred, "test")

            input, target, pred = None, None, None
            if controls is not None:
                controls = None
            torch.cuda.empty_cache()

        return loss


# -----------------------------------------------------------------------------
# Black Box System with truncated back-propagation through time
# and gradient update every N samples
# -----------------------------------------------------------------------------


class BlackBoxSystemWithTBPTT(BlackBoxSystem):
    def __init__(
        self,
        model: BlackBoxModel,
        loss: torch.nn.Module,
        lr: float = 1e-4,
        log_media_every_n_steps: int = 10000,
        step_num_samples: int = 2048,
        use_callbacks: bool = False,
    ):
        super().__init__(model, loss, lr, log_media_every_n_steps, use_callbacks)
        self.step_num_samples = step_num_samples
        self.automatic_optimization = False  # disables lightning optimization
        self.apply_gradient_clipping = False
        self.gradient_clip_val = 1.0
        self.gradient_clip_algorithm = "norm"

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        mode: str,
    ):
        train = True if mode == "train" else False

        # reset hidden states for recurrent layers
        self.model.reset_states()
        if mode == "train":
            self.model.detach_states()
            optimizer = self.optimizers()
            optimizer.zero_grad()

        # get batch
        if self.model.num_controls > 0:
            input, target, controls = batch
        else:
            input, target = batch
            controls = None

        # run the model in steps
        seq_len = input.size(-1)
        pred_chunks = []

        for start_idx in range(0, seq_len, self.step_num_samples):
            end_idx = start_idx + self.step_num_samples
            if end_idx > seq_len:
                end_idx = seq_len

            step_input = input[:, :, start_idx:end_idx]
            step_target = target[:, :, start_idx:end_idx]
            step_pred = self(step_input, controls, train=train)

            pred_chunks.append(step_pred)

            if mode == "train":
                tot_step_loss = self.compute_and_log_loss(step_pred, step_target, mode, should_log=False)
                self.manual_backward(tot_step_loss)
                # need to apply gradient clipping manually
                if self.apply_gradient_clipping:
                    self.clip_gradients(optimizer, self.gradient_clip_val, self.gradient_clip_algorithm)
                optimizer.step()
                self.model.detach_states()
                optimizer.zero_grad()

        pred = torch.cat(pred_chunks, dim=-1)
        assert input.shape == target.shape == pred.shape

        # calculate loss
        tot_batch_loss = self.compute_and_log_loss(pred, target, mode, should_log=True)

        # calculate metrics
        self.compute_and_log_metrics(pred, target, mode)
        return tot_batch_loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="val")
        # log media
        if batch_idx == 0:
            if (self.trainer.global_step / self.log_media_every_n_steps) > self.log_media_counter:
                if self.model.num_controls > 0:
                    input, target, controls = batch
                else:
                    input, target = batch
                    controls = None

                # run the model in steps
                seq_len = input.size(-1)
                pred_chunks = []

                for start_idx in range(0, seq_len, self.step_num_samples):
                    end_idx = start_idx + self.step_num_samples
                    if end_idx > seq_len:
                        end_idx = seq_len

                    step_input = input[:, :, start_idx:end_idx]
                    step_target = target[:, :, start_idx:end_idx]
                    step_pred = self(step_input, controls)

                    pred_chunks.append(step_pred)

                pred = torch.cat(pred_chunks, dim=-1)
                assert input.shape == target.shape == pred.shape
                input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

                self.log_audio(batch_idx, input, target, pred, "val")
                self.log_media_counter += 1

                input, target, pred = None, None, None
                if controls is not None:
                    controls = None
        return loss

    def on_validation_epoch_end(self):
        self.lr_schedulers().step(self.trainer.logged_metrics["loss/val/tot"])


# -----------------------------------------------------------------------------
# Grey Box System with truncated back-propagation through time
# at the start of each batch
# -----------------------------------------------------------------------------


class GreyBoxSystem(BaseSystem):
    def __init__(
        self,
        model: GreyBoxModel,
        loss: torch.nn.Module,
        lr: float = 1e-4,
        log_media_every_n_steps: int = 10000,
        use_callbacks: bool = False,
    ):
        super().__init__(loss, lr, log_media_every_n_steps, use_callbacks)
        self.model = model

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        mode: str,
    ):
        """Model step used for validation and training.
        Args:
            batch (Tuple[Tensor, Tensor, Tensor]): Batch items containing input audio (x) target audio (y), and noise profile (w).
            batch_idx (int): Index of the batch within the current epoch.
            mode (str): One of "train", "val", "test".
        """
        train = True if mode == "train" else False

        # reset hidden states for recurrent layers
        self.model.reset_states()
        if mode == "train":
            self.model.detach_states()

        # get batch
        if self.model.num_controls > 0:
            input, target, controls = batch
        else:
            input, target = batch
            controls = None

        # run the model
        pred = self(input, controls, train=train)

        # calculate loss
        tot_batch_loss = self.compute_and_log_loss(pred, target, mode)

        # calculate metrics
        self.compute_and_log_metrics(pred, target, mode)

        return tot_batch_loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="val")
        # log media
        if batch_idx == 0:
            if (self.trainer.global_step / self.log_media_every_n_steps) > self.log_media_counter:
                if self.model.num_controls > 0:
                    input, target, controls = batch
                else:
                    input, target = batch
                    controls = None

                self.model.eval()
                self.model.reset_states()
                self.model.detach_states()
                with torch.no_grad():
                    pred = self(input, controls)
                self.model.train()
                input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

                self.log_audio(batch_idx, input, target, pred, "val")
                # self.log_audio_at_each_block(input, controls)
                self.log_response_and_params_at_each_block(input, controls, "val")
                self.log_media_counter += 1

                input, target, pred = None, None, None
                if controls is not None:
                    controls = None
                torch.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="test")

        # log media
        if self.model.num_controls > 0:
            input, target, controls = batch
        else:
            input, target = batch
            controls = None

        self.model.reset_states()
        with torch.no_grad():
            pred = self(input, controls)
        input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

        if batch_idx < 10:
            self.log_audio(batch_idx, input, target, pred, "test")
            self.log_response_and_params_at_each_block(input, controls, "test")

        input, target, pred = None, None, None
        if controls is not None:
            controls = None
        torch.cuda.empty_cache()

        return loss

    def on_train_start(self):
        # Parameter visualization handled by callbacks if use_callbacks=True
        if not self.use_callbacks:
            batch = next(iter(self.trainer.train_dataloader))
            if self.model.num_controls > 0:
                input, target, controls = batch
                controls = controls.to(self.device)
            else:
                input, target = batch
                controls = None
            input = input.to(self.device)
            target = target.to(self.device)
            self.model.reset_states()
            self.log_response_and_params_at_each_block(input, controls, "val")
            # self.log_frequency_response() # atm needs too much memory

    def log_audio_at_each_block(self, input, controls):
        print("\nLogging audio at each block...")
        x = input.to(self.device)
        y = [input]
        control_params = self.model.controller(x, controls if controls is not None else None)
        for prc, ctrl_prms in zip(self.model.processor.processors, control_params):
            if ctrl_prms is not None:
                ctrl_prms = ctrl_prms.to(self.device)
            y_i, _ = prc(x, ctrl_prms, train=False)
            y.append(y_i)
            x = y[-1]

        for i, _ in enumerate(y):  # for each block in the chain
            for j, _ in enumerate(y[i]):  # for each item in the batch
                self.logger.experiment.log(
                    {f"audio/chain/{j}/block{i}": [wandb.Audio(y[i][j].cpu().numpy()[0, :], 48000, caption=f"pred_{i}_block{j}")]},
                    step=self.trainer.global_step,
                )

    def log_response_and_params_at_each_block(self, input, controls, mode="val"):
        """Log response and parameters at each block. No-op if using callbacks."""
        if self.use_callbacks:
            return  # Parameter visualization handled by ParameterVisualizationCallback

        print("\nLogging response and parameters at each block...")
        x = input.to(self.device)
        control_params = self.model.controller(x, controls if controls is not None else None)
        param_dict_list = []
        for prc, ctrl_prms in zip(self.model.processor.processors, control_params):
            if ctrl_prms is not None:
                ctrl_prms = ctrl_prms.to(self.device)
            _, param_dict = prc(x, ctrl_prms, train=False)
            param_dict_list.append(param_dict)

        for i, _ in enumerate(input):
            plot = plot_gb_model(self.model, param_dict_list, input, i)
            self.logger.experiment.log(
                {f"response_blocks/{mode}/{i}": [wandb.Image(plot, caption=f"response_{i}")]}, step=self.trainer.global_step
            )

    def configure_optimizers(self):
        parameters = []
        # include all parameters in the model with their respective learning rates
        print("\nPer layer learning rates:")
        for module in self.model.processor.processors:
            print(type(module))
            for idx, (name, param) in enumerate(module.named_parameters()):
                # append layer parameters
                print(idx, name, self.lr * module.lr_multiplier)
                parameters += [
                    {
                        "params": [p for n, p in module.named_parameters() if n == name and p.requires_grad],
                        "lr": self.lr * module.lr_multiplier,
                    }
                ]
        print()
        for module in self.model.controller.controllers:
            print(type(module))
            for idx, (name, param) in enumerate(module.named_parameters()):
                # append layer parameters
                print(idx, name, self.lr * module.lr_multiplier)
                parameters += [
                    {
                        "params": [p for n, p in module.named_parameters() if n == name and p.requires_grad],
                        "lr": self.lr * module.lr_multiplier,
                    }
                ]

        optimizer = torch.optim.AdamW(
            parameters,
            # lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)

        return [optimizer], [{"scheduler": lr_scheduler, "monitor": "loss/val/tot", "interval": "epoch", "frequency": 1}]


# -----------------------------------------------------------------------------
# Grey Box System with truncated back-propagation through time
# and gradient update every N samples
# -----------------------------------------------------------------------------


class GreyBoxSystemWithTBPTT(GreyBoxSystem):
    def __init__(
        self,
        model: GreyBoxModel,
        loss: torch.nn.Module,
        lr: float = 1e-4,
        log_media_every_n_steps: int = 10000,
        step_num_samples: int = 2048,
    ):
        super().__init__(model, loss, lr, log_media_every_n_steps)
        self.step_num_samples = step_num_samples
        self.automatic_optimization = False  # disables lightning optimization
        self.apply_gradient_clipping = False
        self.gradient_clip_val = 2.0
        self.gradient_clip_algorithm = "norm"

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        mode: str,
    ):
        train = True if mode == "train" else False

        # reset hidden states for recurrent layers
        self.model.reset_states()
        if mode == "train":
            self.model.detach_states()
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.zero_grad()

        # get batch
        if self.model.num_controls > 0:
            input, target, controls = batch
        else:
            input, target = batch
            controls = None

        # run the model in steps
        seq_len = input.size(-1)
        pred_chunks = []

        for start_idx in range(0, seq_len, self.step_num_samples):
            end_idx = start_idx + self.step_num_samples
            if end_idx > seq_len:
                end_idx = seq_len

            step_input = input[:, :, start_idx:end_idx]
            step_target = target[:, :, start_idx:end_idx]
            step_pred = self(step_input, controls, train=train)

            pred_chunks.append(step_pred)

            if mode == "train":
                tot_step_loss = self.compute_and_log_loss(step_pred, step_target, mode, should_log=False)
                self.manual_backward(tot_step_loss)
                # need to apply gradient clipping manually
                if self.apply_gradient_clipping:
                    self.clip_gradients(optimizer, self.gradient_clip_val, self.gradient_clip_algorithm)
                optimizer.step()
                self.model.detach_states()
                optimizer.zero_grad()

        pred = torch.cat(pred_chunks, dim=-1)
        assert input.shape == target.shape == pred.shape

        # calculate loss
        tot_batch_loss = self.compute_and_log_loss(pred, target, mode, should_log=True)

        # calculate metrics
        self.compute_and_log_metrics(pred, target, mode)

        return tot_batch_loss

    def on_validation_epoch_end(self):
        self.lr_schedulers().step(self.trainer.logged_metrics["loss/val/tot"])
