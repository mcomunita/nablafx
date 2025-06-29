"""
Audio Logging Callback

Handles logging of audio samples during training and validation.
Extracted from system.py to provide modular, configurable audio logging.
"""

import torch
import wandb
from typing import Any, Optional
import lightning as pl
from lightning.pytorch.callbacks import Callback


class AudioLoggingCallback(Callback):
    """
    Callback for logging audio samples during training.

    This callback handles the logging of input, target, and predicted audio samples
    to Weights & Biases during training and testing phases.

    Args:
        log_every_n_steps: Log audio every N global steps during validation
        sample_rate: Sample rate for audio logging (default: 48000)
        max_samples_per_batch: Maximum number of samples to log per batch (default: 5)
        log_test_batches: Number of test batches to log (default: 10)
        log_input_target_once: Whether to log input/target only once during validation (default: True)
    """

    def __init__(
        self,
        log_every_n_steps: int = 10000,
        sample_rate: int = 48000,
        max_samples_per_batch: int = 5,
        log_test_batches: int = 10,
        log_input_target_once: bool = True,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.sample_rate = sample_rate
        self.max_samples_per_batch = max_samples_per_batch
        self.log_test_batches = log_test_batches
        self.log_input_target_once = log_input_target_once

        # Internal state
        self.log_media_counter = 0
        self.log_input_and_target_flag = True

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log audio during validation if conditions are met."""
        if batch_idx == 0:  # Only log first batch
            if (trainer.global_step / self.log_every_n_steps) > self.log_media_counter:
                self._log_validation_audio(trainer, pl_module, batch, batch_idx)
                self.log_media_counter += 1

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log audio during testing."""
        if batch_idx < self.log_test_batches:
            self._log_test_audio(trainer, pl_module, batch, batch_idx)

    def _log_validation_audio(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Generate and log validation audio."""
        # Extract batch data
        if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
            input_audio, target_audio, controls = batch
        else:
            input_audio, target_audio = batch
            controls = None

        # Generate predictions
        pl_module.model.reset_states()
        if hasattr(pl_module.model, "detach_states"):
            pl_module.model.detach_states()

        with torch.no_grad():
            pred_audio = pl_module(input_audio, controls)

        # Move to CPU for logging
        input_audio = input_audio.detach().cpu()
        target_audio = target_audio.detach().cpu()
        pred_audio = pred_audio.detach().cpu()

        # Log audio samples
        self._log_audio_samples(trainer, input_audio, target_audio, pred_audio, batch_idx, "val")

        # Cleanup
        del input_audio, target_audio, pred_audio
        if controls is not None:
            del controls

    def _log_test_audio(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Generate and log test audio."""
        # Extract batch data
        if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
            input_audio, target_audio, controls = batch
        else:
            input_audio, target_audio = batch
            controls = None

        # Generate predictions
        pl_module.model.reset_states()
        with torch.no_grad():
            pred_audio = pl_module(input_audio, controls)

        # Move to CPU for logging
        input_audio = input_audio.detach().cpu()
        target_audio = target_audio.detach().cpu()
        pred_audio = pred_audio.detach().cpu()

        # Log audio samples
        self._log_audio_samples(trainer, input_audio, target_audio, pred_audio, batch_idx, "test")

        # Cleanup
        del input_audio, target_audio, pred_audio
        if controls is not None:
            del controls
        torch.cuda.empty_cache()

    def _log_audio_samples(
        self,
        trainer: pl.Trainer,
        input_audio: torch.Tensor,
        target_audio: torch.Tensor,
        pred_audio: torch.Tensor,
        batch_idx: int,
        mode: str,
    ) -> None:
        """Log audio samples to wandb."""
        if not hasattr(trainer.logger, "experiment"):
            return  # Skip if no wandb logger

        num_samples = min(len(input_audio), self.max_samples_per_batch)

        for i in range(num_samples):
            # Prepare audio data (ensure correct format)
            input_np = input_audio[i].numpy()
            target_np = target_audio[i].numpy()
            pred_np = pred_audio[i].numpy()

            # Handle different tensor shapes
            if input_np.ndim > 1:
                input_np = input_np[0, :]  # Take first channel
            if target_np.ndim > 1:
                target_np = target_np[0, :]
            if pred_np.ndim > 1:
                pred_np = pred_np[0, :]

            log_dict = {}

            # Log based on mode and settings
            if mode == "test":
                # Log all audio for test mode
                log_dict.update(
                    {
                        f"audio/{mode}/input/b{batch_idx}-{i}": wandb.Audio(input_np, self.sample_rate, caption=f"input_b{batch_idx}-{i}"),
                        f"audio/{mode}/target/b{batch_idx}-{i}": wandb.Audio(
                            target_np, self.sample_rate, caption=f"target_b{batch_idx}-{i}"
                        ),
                        f"audio/{mode}/pred/b{batch_idx}-{i}": wandb.Audio(pred_np, self.sample_rate, caption=f"pred_b{batch_idx}-{i}"),
                    }
                )
            elif self.log_input_target_once and self.log_input_and_target_flag:
                # Log input, target, and prediction once
                log_dict.update(
                    {
                        f"audio/{mode}/input/b{batch_idx}-{i}": wandb.Audio(input_np, self.sample_rate, caption=f"input_b{batch_idx}-{i}"),
                        f"audio/{mode}/target/b{batch_idx}-{i}": wandb.Audio(
                            target_np, self.sample_rate, caption=f"target_b{batch_idx}-{i}"
                        ),
                        f"audio/{mode}/pred/b{batch_idx}-{i}": wandb.Audio(pred_np, self.sample_rate, caption=f"pred_b{batch_idx}-{i}"),
                    }
                )
                if i == num_samples - 1:  # Set flag after logging all samples
                    self.log_input_and_target_flag = False
            else:
                # Log only predictions
                log_dict.update(
                    {
                        f"audio/{mode}/pred/b{batch_idx}-{i}": wandb.Audio(pred_np, self.sample_rate, caption=f"pred_b{batch_idx}-{i}"),
                    }
                )

            # Log to wandb
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
