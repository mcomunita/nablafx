"""
Metrics Logging Callback

Handles computation and logging of various metrics during training.
Extracted from system.py to provide modular, configurable metrics logging.
"""

import torch
import torchmetrics as tm
import auraloss
from typing import Any, Dict, Optional
import lightning as pl
from lightning.pytorch.callbacks import Callback


class MetricsLoggingCallback(Callback):
    """
    Callback for computing and logging metrics during training.

    This callback handles the computation and logging of various audio metrics
    such as MAE, MSE, ESR, etc. during training and validation.

    Args:
        metrics_config: Dictionary defining which metrics to compute.
                       If None, uses default set of metrics.
        log_on_step: Whether to log metrics on each step (default: False)
        log_on_epoch: Whether to log metrics on each epoch (default: True)
        sync_dist: Whether to sync metrics across distributed processes (default: True)
    """

    def __init__(
        self,
        metrics_config: Optional[Dict[str, Any]] = None,
        log_on_step: bool = False,
        log_on_epoch: bool = True,
        sync_dist: bool = True,
    ):
        super().__init__()
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.sync_dist = sync_dist

        # Initialize metrics
        self.metrics = self._create_metrics(metrics_config)

    def _create_metrics(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, torch.nn.Module]:
        """Create metrics based on configuration."""
        if config is None:
            # Default metrics (matching original system.py)
            return {
                "mae": tm.MeanAbsoluteError(),
                "mape": tm.MeanAbsolutePercentageError(),
                "mse": tm.MeanSquaredError(),
                "cossim": tm.CosineSimilarity(),
                "logcosh": auraloss.time.LogCoshLoss(),
                "esr": auraloss.time.ESRLoss(),
                "dcloss": auraloss.time.DCLoss(),
            }

        # Create metrics from config
        metrics = {}
        for name, metric_config in config.items():
            if isinstance(metric_config, str):
                # Simple string reference
                metrics[name] = self._get_metric_by_name(metric_config)
            elif isinstance(metric_config, dict):
                # Dictionary config with class_path and init_args
                metrics[name] = self._create_metric_from_config(metric_config)

        return metrics

    def _get_metric_by_name(self, name: str) -> torch.nn.Module:
        """Get metric by name string."""
        metric_registry = {
            "mae": tm.MeanAbsoluteError,
            "mape": tm.MeanAbsolutePercentageError,
            "mse": tm.MeanSquaredError,
            "cossim": tm.CosineSimilarity,
            "logcosh": auraloss.time.LogCoshLoss,
            "esr": auraloss.time.ESRLoss,
            "dcloss": auraloss.time.DCLoss,
        }

        if name not in metric_registry:
            raise ValueError(f"Unknown metric: {name}")

        return metric_registry[name]()

    def _create_metric_from_config(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Create metric from configuration dictionary."""
        # This is a simplified version - in practice you'd want more robust
        # dynamic class loading similar to the loss system
        class_path = config.get("class_path", "")
        init_args = config.get("init_args", {})

        # For now, handle common cases
        if "torchmetrics" in class_path:
            # Handle torchmetrics
            module_name = class_path.split(".")[-1]
            if hasattr(tm, module_name):
                return getattr(tm, module_name)(**init_args)
        elif "auraloss" in class_path:
            # Handle auraloss metrics
            parts = class_path.split(".")
            if len(parts) >= 3:  # e.g., auraloss.time.ESRLoss
                domain = parts[1]  # time or freq
                metric_name = parts[2]
                domain_module = getattr(auraloss, domain)
                return getattr(domain_module, metric_name)(**init_args)

        raise ValueError(f"Cannot create metric from config: {config}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Compute metrics on training batch end."""
        if self.log_on_step:
            self._compute_and_log_metrics(trainer, pl_module, batch, "train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute metrics on validation batch end."""
        self._compute_and_log_metrics(trainer, pl_module, batch, "val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute metrics on test batch end."""
        self._compute_and_log_metrics(trainer, pl_module, batch, "test")

    def _compute_and_log_metrics(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        mode: str,
    ) -> None:
        """Compute and log metrics for the given batch."""
        # Extract batch data
        if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
            input_audio, target = batch[:2]
            controls = batch[2] if len(batch) > 2 else None
        else:
            input_audio, target = batch[:2]
            controls = None

        # Generate predictions
        pl_module.model.reset_states()
        if mode == "train" and hasattr(pl_module.model, "detach_states"):
            pl_module.model.detach_states()

        with torch.no_grad():
            pred = pl_module(input_audio, controls, train=(mode == "train"))

        # Compute metrics
        for name, metric in self.metrics.items():
            try:
                # Move metric to correct device
                metric = metric.to(pl_module.device)

                # Prepare tensors
                pred_for_metric = pred.clone()
                target_for_metric = target.clone()

                # Handle tensor dimensions (some metrics expect 2D)
                if pred_for_metric.dim() == 3:
                    pred_for_metric = pred_for_metric.squeeze(1)
                    target_for_metric = target_for_metric.squeeze(1)

                # Compute metric
                metric_value = metric(pred_for_metric, target_for_metric)

                # Clean up tensors
                pred_for_metric = pred_for_metric.detach().cpu()
                target_for_metric = target_for_metric.detach().cpu()
                metric_value = metric_value.detach().cpu()

                # Log metric
                pl_module.log(
                    f"metrics/{mode}/{name}",
                    metric_value,
                    on_step=self.log_on_step,
                    on_epoch=self.log_on_epoch,
                    prog_bar=False,
                    logger=True,
                    sync_dist=self.sync_dist,
                )

                # Reset metric state if possible
                if hasattr(metric, "reset"):
                    metric.reset()

            except Exception as e:
                # Log error but don't crash training
                print(f"Warning: Failed to compute metric {name}: {e}")
                continue

        # Cleanup
        del pred, target, input_audio
        if controls is not None:
            del controls
