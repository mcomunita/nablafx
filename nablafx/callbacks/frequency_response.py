"""
Frequency Response Callback

Handles logging of frequency response plots during training.
Extracted from system.py to provide modular, configurable frequency response logging.
"""

import torch
import wandb
from typing import Any, Optional
import lightning as pl
from lightning.pytorch.callbacks import Callback
from nablafx.plotting import plot_frequency_response_steps


class FrequencyResponseCallback(Callback):
    """
    Callback for logging frequency response plots.

    This callback handles the generation and logging of frequency response plots
    during training to visualize model behavior across different frequencies.

    Args:
        log_on_train_start: Whether to log frequency response at training start (default: False)
        log_on_train_end: Whether to log frequency response at training end (default: False)
        log_on_test_end: Whether to log frequency response at test end (default: True)
        log_every_n_epochs: Log frequency response every N epochs during validation (default: None)
        frequency_range: Tuple of (min_freq, max_freq) for frequency analysis (default: None)
    """

    def __init__(
        self,
        log_on_train_start: bool = False,
        log_on_train_end: bool = False,
        log_on_test_end: bool = True,
        log_every_n_epochs: Optional[int] = None,
        frequency_range: Optional[tuple] = None,
    ):
        super().__init__()
        self.log_on_train_start = log_on_train_start
        self.log_on_train_end = log_on_train_end
        self.log_on_test_end = log_on_test_end
        self.log_every_n_epochs = log_every_n_epochs
        self.frequency_range = frequency_range

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log frequency response at training start."""
        if self.log_on_train_start:
            self._log_frequency_response(trainer, pl_module, "train_start")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log frequency response at training end."""
        if self.log_on_train_end:
            self._log_frequency_response(trainer, pl_module, "train_end")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log frequency response at test end."""
        if self.log_on_test_end:
            self._log_frequency_response(trainer, pl_module, "test")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log frequency response every N validation epochs."""
        if self.log_every_n_epochs is not None and trainer.current_epoch % self.log_every_n_epochs == 0:
            self._log_frequency_response(trainer, pl_module, "val")

    def _log_frequency_response(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        phase: str,
    ) -> None:
        """Generate and log frequency response plot."""
        if not hasattr(trainer.logger, "experiment"):
            return  # Skip if no wandb logger

        try:
            print(f"\nLogging frequency response for {phase}...")

            # Reset model states
            pl_module.model.reset_states()

            # Generate frequency response plot
            with torch.no_grad():
                plot = plot_frequency_response_steps(pl_module.model)

            # Log to wandb
            trainer.logger.experiment.log(
                {f"response/freq+phase/{phase}": wandb.Image(plot, caption=f"Frequency Response - {phase}")}, step=trainer.global_step
            )

            print(f"Successfully logged frequency response for {phase}")

        except Exception as e:
            print(f"Warning: Failed to log frequency response for {phase}: {e}")
            # Don't crash training if frequency response logging fails
