"""
Simplified System Classes for Callback-Based Architecture

This demonstrates how the system classes can be simplified when using the new
callback system. The logging logic has been extracted to callbacks, making
the system classes focused on core training logic.
"""

import torch
import lightning as pl
from typing import Any
from nablafx.models import BlackBoxModel, GreyBoxModel


class BaseSystemWithCallbacks(pl.LightningModule):
    """
    Simplified base system that works with the callback architecture.

    All logging functionality has been moved to callbacks, making this class
    focused purely on the core training logic.
    """

    def __init__(self, loss: torch.nn.Module, lr: float = 1e-4):
        super().__init__()
        self.loss = loss
        self.lr = lr

        # Save hyperparameters for Lightning
        self.save_hyperparameters(ignore=["loss"])

    def forward(self, input_audio: torch.Tensor, params: torch.Tensor, train: bool = False):
        return self.model(input_audio, params, train=train)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss without logging (callbacks handle logging).
        Returns only the total loss for backpropagation.
        """
        losses = self.loss(pred, target)

        # Handle different loss function formats
        if hasattr(self.loss, "get_loss_names"):  # WeightedMultiLoss
            if isinstance(losses, tuple):
                return losses[-1]  # Return total loss
            else:
                return losses
        else:
            # Simple loss or other formats
            if isinstance(losses, (tuple, list)):
                return sum(losses)
            return losses

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="test")

    def common_step(self, batch, batch_idx, mode="train"):
        """Simplified common step - callbacks handle logging."""
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss/val/tot",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class BlackBoxSystemWithCallbacks(BaseSystemWithCallbacks):
    """
    Simplified BlackBox system that works with callbacks.

    All logging is handled by callbacks, making this class much cleaner.
    """

    def __init__(
        self,
        model: BlackBoxModel,
        loss: torch.nn.Module,
        lr: float = 1e-4,
    ):
        super().__init__(loss, lr)
        self.model = model

    def common_step(self, batch, batch_idx, mode="train"):
        """Simplified step without logging - callbacks handle everything."""
        train = mode == "train"

        # Reset hidden states for recurrent layers
        self.model.reset_states()
        if train and hasattr(self.model, "detach_states"):
            self.model.detach_states()

        # Extract batch data
        if hasattr(self.model, "num_controls") and self.model.num_controls > 0:
            input_audio, target, controls = batch
        else:
            input_audio, target = batch
            controls = None

        # Forward pass
        pred = self(input_audio, controls, train=train)

        # Compute loss
        loss = self.compute_loss(pred, target)

        # Log the loss (this is the minimal logging we keep in the system)
        self.log(
            f"loss/{mode}/tot",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss


class GreyBoxSystemWithCallbacks(BaseSystemWithCallbacks):
    """
    Simplified GreyBox system that works with callbacks.

    Parameter logging is handled by ParameterVisualizationCallback.
    """

    def __init__(
        self,
        model: GreyBoxModel,
        loss: torch.nn.Module,
        lr: float = 1e-4,
    ):
        super().__init__(loss, lr)
        self.model = model

    def common_step(self, batch, batch_idx, mode="train"):
        """Simplified step without logging - callbacks handle everything."""
        train = mode == "train"

        # Reset hidden states for recurrent layers
        self.model.reset_states()
        if train and hasattr(self.model, "detach_states"):
            self.model.detach_states()

        # Extract batch data
        if hasattr(self.model, "num_controls") and self.model.num_controls > 0:
            input_audio, target, controls = batch
        else:
            input_audio, target = batch
            controls = None

        # Forward pass
        pred = self(input_audio, controls, train=train)

        # Compute loss
        loss = self.compute_loss(pred, target)

        # Log the loss (minimal logging kept in system)
        self.log(
            f"loss/{mode}/tot",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        """Custom optimizer setup for gray-box models with per-layer learning rates."""
        parameters = []

        print("\nPer layer learning rates:")

        # Processor parameters
        for module in self.model.processor.processors:
            print(f"Processor: {type(module)}")
            for idx, (name, param) in enumerate(module.named_parameters()):
                lr_multiplier = getattr(module, "lr_multiplier", 1.0)
                print(f"  {idx}: {name} -> lr={self.lr * lr_multiplier}")
                parameters.append(
                    {
                        "params": [param],
                        "lr": self.lr * lr_multiplier,
                    }
                )

        # Controller parameters
        for module in self.model.controller.controllers:
            print(f"Controller: {type(module)}")
            for idx, (name, param) in enumerate(module.named_parameters()):
                lr_multiplier = getattr(module, "lr_multiplier", 1.0)
                print(f"  {idx}: {name} -> lr={self.lr * lr_multiplier}")
                parameters.append(
                    {
                        "params": [param],
                        "lr": self.lr * lr_multiplier,
                    }
                )

        optimizer = torch.optim.AdamW(
            parameters,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss/val/tot",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# Backward compatibility aliases (optional)
BlackBoxSystem = BlackBoxSystemWithCallbacks
GreyBoxSystem = GreyBoxSystemWithCallbacks
