"""
Parameter Visualization Callback

Handles visualization of gray-box model parameters during training.
Extracted from system.py to provide modular, configurable parameter visualization.
"""

import torch
import wandb
from typing import Any, Optional
import lightning as pl
from lightning.pytorch.callbacks import Callback
from nablafx.plotting import plot_gb_model


class ParameterVisualizationCallback(Callback):
    """
    Callback for visualizing gray-box model parameters.

    This callback handles the visualization of parameters and frequency responses
    for each block in gray-box models during training and testing.

    Args:
        log_on_train_start: Whether to log parameters at training start (default: True)
        log_on_validation: Whether to log parameters during validation (default: True)
        log_on_test: Whether to log parameters during testing (default: True)
        log_test_batches: Number of test batches to visualize (default: 10)
        max_samples_per_batch: Maximum samples to visualize per batch (default: 5)
    """

    def __init__(
        self,
        log_on_train_start: bool = True,
        log_on_validation: bool = True,
        log_on_test: bool = True,
        log_test_batches: int = 10,
        max_samples_per_batch: int = 5,
    ):
        super().__init__()
        self.log_on_train_start = log_on_train_start
        self.log_on_validation = log_on_validation
        self.log_on_test = log_on_test
        self.log_test_batches = log_test_batches
        self.max_samples_per_batch = max_samples_per_batch

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log parameters at training start."""
        if self.log_on_train_start and self._is_greybox_model(pl_module):
            try:
                # Get a batch from the dataloader
                batch = next(iter(trainer.train_dataloader))

                # Move batch to device
                if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
                    input_audio, target, controls = batch
                    controls = controls.to(pl_module.device)
                else:
                    input_audio, target = batch
                    controls = None

                input_audio = input_audio.to(pl_module.device)
                target = target.to(pl_module.device)

                # Log parameters
                pl_module.model.reset_states()
                self._log_response_and_params_at_each_block(trainer, pl_module, input_audio, controls, "train_start")

            except Exception as e:
                print(f"Warning: Failed to log parameters at train start: {e}")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log parameters during validation."""
        if self.log_on_validation and batch_idx == 0 and self._is_greybox_model(pl_module):

            # Extract batch data
            if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
                input_audio, target, controls = batch
            else:
                input_audio, target = batch
                controls = None

            self._log_response_and_params_at_each_block(trainer, pl_module, input_audio, controls, "val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log parameters during testing."""
        if self.log_on_test and batch_idx < self.log_test_batches and self._is_greybox_model(pl_module):

            # Extract batch data
            if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
                input_audio, target, controls = batch
            else:
                input_audio, target = batch
                controls = None

            self._log_response_and_params_at_each_block(trainer, pl_module, input_audio, controls, "test")

    def _is_greybox_model(self, pl_module: pl.LightningModule) -> bool:
        """Check if the model is a gray-box model."""
        return (
            hasattr(pl_module.model, "processor")
            and hasattr(pl_module.model, "controller")
            and hasattr(pl_module.model.processor, "processors")
            and hasattr(pl_module.model.controller, "controllers")
        )

    def _log_response_and_params_at_each_block(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        input_audio: torch.Tensor,
        controls: Optional[torch.Tensor],
        mode: str,
    ) -> None:
        """Log frequency response and parameters for each block."""
        if not hasattr(trainer.logger, "experiment"):
            return  # Skip if no wandb logger

        try:
            print(f"\nLogging response and parameters at each block for {mode}...")

            # Get control parameters for each processor
            control_params = pl_module.model.controller(input_audio, controls if controls is not None else None)

            # Get parameter dictionaries from each processor
            param_dict_list = []
            x = input_audio

            for processor, ctrl_params in zip(pl_module.model.processor.processors, control_params):
                if ctrl_params is not None:
                    ctrl_params = ctrl_params.to(pl_module.device)

                # Get parameters without running the full forward pass
                with torch.no_grad():
                    _, param_dict = processor(x, ctrl_params, train=False)

                param_dict_list.append(param_dict)

            # Generate and log plots for each sample in the batch
            num_samples = min(len(input_audio), self.max_samples_per_batch)

            for i in range(num_samples):
                try:
                    plot = plot_gb_model(pl_module.model, param_dict_list, input_audio, i)

                    trainer.logger.experiment.log(
                        {f"response_blocks/{mode}/{i}": wandb.Image(plot, caption=f"response_{i}")}, step=trainer.global_step
                    )

                except Exception as e:
                    print(f"Warning: Failed to create plot for sample {i}: {e}")
                    continue

            print(f"Successfully logged parameter visualization for {mode}")

        except Exception as e:
            print(f"Warning: Failed to log response and parameters for {mode}: {e}")

    def log_audio_at_each_block(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        input_audio: torch.Tensor,
        controls: Optional[torch.Tensor],
        mode: str = "debug",
    ) -> None:
        """
        Log audio output at each block in the processing chain.

        This method can be called manually for debugging purposes.
        """
        if not hasattr(trainer.logger, "experiment"):
            return  # Skip if no wandb logger

        try:
            print(f"\nLogging audio at each block for {mode}...")

            x = input_audio.to(pl_module.device)
            y = [input_audio]  # Store audio at each stage

            # Get control parameters
            control_params = pl_module.model.controller(x, controls if controls is not None else None)

            # Process through each block
            for processor, ctrl_params in zip(pl_module.model.processor.processors, control_params):
                if ctrl_params is not None:
                    ctrl_params = ctrl_params.to(pl_module.device)

                with torch.no_grad():
                    y_i, _ = processor(x, ctrl_params, train=False)

                y.append(y_i)
                x = y[-1]

            # Log audio for each block and each batch item
            for block_idx, block_audio in enumerate(y):
                for batch_idx in range(len(block_audio)):
                    audio_np = block_audio[batch_idx].cpu().numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np[0, :]  # Take first channel

                    trainer.logger.experiment.log(
                        {
                            f"audio/chain/{batch_idx}/block{block_idx}": wandb.Audio(
                                audio_np, 48000, caption=f"pred_{block_idx}_block{batch_idx}"
                            )
                        },
                        step=trainer.global_step,
                    )

            print(f"Successfully logged audio at each block for {mode}")

        except Exception as e:
            print(f"Warning: Failed to log audio at each block for {mode}: {e}")
