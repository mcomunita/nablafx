"""
Test script demonstrating the new callback system.

This script shows how to use the callback-based logging architecture
instead of the hardcoded logging in system.py.
"""

import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Import the new callbacks
from nablafx.callbacks import (
    AudioLoggingCallback,
    MetricsLoggingCallback,
    FrequencyResponseCallback,
    FADComputationCallback,
    ParameterVisualizationCallback,
)

# For this example, we'll use the existing system but show the callback setup
from nablafx.system import BlackBoxSystem
from nablafx.models import BlackBoxModel
from nablafx.loss import WeightedMultiLoss
from nablafx.data import DataModule  # Assuming this exists


def create_callback_based_trainer():
    """
    Create a trainer with the new callback system.

    This demonstrates how to replace the hardcoded logging in system.py
    with configurable callbacks.
    """

    # Create the callbacks
    callbacks = [
        # Audio logging - replaces log_audio() method
        AudioLoggingCallback(
            log_every_n_steps=1000,
            sample_rate=48000,
            max_samples_per_batch=3,
            log_test_batches=5,
            log_input_target_once=True,
        ),
        # Metrics logging - replaces compute_and_log_metrics() method
        MetricsLoggingCallback(
            log_on_step=False,
            log_on_epoch=True,
            sync_dist=True,
        ),
        # Frequency response - replaces log_frequency_response() method
        FrequencyResponseCallback(
            log_on_train_start=False,
            log_on_train_end=False,
            log_on_test_end=True,
            log_every_n_epochs=5,
        ),
        # FAD computation - replaces compute_and_log_fad() method
        FADComputationCallback(
            compute_on_train_end=True,
            compute_on_test_end=True,
            compute_every_n_epochs=10,
            models=["vggish", "pann"],  # Start with fewer models for testing
            checkpoint_dir="checkpoints_fad",
        ),
        # Parameter visualization - for gray-box models only
        # ParameterVisualizationCallback(
        #     log_on_train_start=True,
        #     log_on_validation=True,
        #     log_on_test=True,
        # ),
        # Standard Lightning callbacks
        ModelCheckpoint(
            monitor="loss/val/tot",
            mode="min",
            save_top_k=3,
            filename="best-{epoch:02d}-{loss/val/tot:.4f}",
        ),
        EarlyStopping(
            monitor="loss/val/tot",
            mode="min",
            patience=50,
            verbose=True,
        ),
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=WandbLogger(project="nablafx-callback-system"),
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )

    return trainer


def create_system_and_data():
    """Create a system and data module for testing."""

    # Create a simple loss function
    loss = WeightedMultiLoss(
        losses=[{"loss": torch.nn.L1Loss(), "weight": 0.5, "name": "l1"}, {"loss": torch.nn.MSELoss(), "weight": 0.5, "name": "mse"}]
    )

    # Create a simple model (you'd replace this with your actual model)
    model = BlackBoxModel(
        # Add your model parameters here
    )

    # Create system
    system = BlackBoxSystem(
        model=model,
        loss=loss,
        lr=1e-4,
        # Note: log_media_every_n_steps is now handled by AudioLoggingCallback
    )

    # Create data module (you'd replace this with your actual data)
    data_module = DataModule(
        # Add your data parameters here
    )

    return system, data_module


def main():
    """Main training function using the new callback system."""

    # Create trainer with callbacks
    trainer = create_callback_based_trainer()

    # Create system and data
    system, data_module = create_system_and_data()

    # Train the model
    trainer.fit(system, data_module)

    # Test the model
    trainer.test(system, data_module)


def demo_yaml_config():
    """
    Demonstrate how to use the callback system with YAML configuration.

    This shows the recommended way to configure callbacks for production use.
    """

    yaml_config = """
# Example trainer config with callbacks
trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1
  
  callbacks:
    # Audio logging callback
    - class_path: nablafx.callbacks.AudioLoggingCallback
      init_args:
        log_every_n_steps: 1000
        sample_rate: 48000
        max_samples_per_batch: 3
        log_test_batches: 5
    
    # Metrics logging callback  
    - class_path: nablafx.callbacks.MetricsLoggingCallback
      init_args:
        log_on_epoch: true
        sync_dist: true
    
    # Frequency response callback
    - class_path: nablafx.callbacks.FrequencyResponseCallback
      init_args:
        log_on_test_end: true
        log_every_n_epochs: 5
    
    # FAD computation callback
    - class_path: nablafx.callbacks.FADComputationCallback
      init_args:
        compute_on_train_end: true
        compute_on_test_end: true
        models: ["vggish", "pann"]
    
    # Standard Lightning callbacks
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: "loss/val/tot"
        mode: "min"
        save_top_k: 3
    
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: "loss/val/tot"
        mode: "min"
        patience: 50

# System config (now much simpler!)
system:
  class_path: nablafx.system.BlackBoxSystem
  init_args:
    lr: 1e-4
    # No more log_media_every_n_steps - handled by callbacks!

# Loss config
loss:
  class_path: nablafx.loss.WeightedMultiLoss
  init_args:
    losses:
      - loss:
          class_path: torch.nn.L1Loss
        weight: 0.5
        name: "l1"
      - loss:
          class_path: torch.nn.MSELoss
        weight: 0.5
        name: "mse"
"""

    print("YAML Configuration Example:")
    print(yaml_config)

    print("\nWith this configuration:")
    print("- All logging is handled by configurable callbacks")
    print("- System classes are much simpler and focused")
    print("- Easy to enable/disable different types of logging")
    print("- Can customize logging behavior per experiment")
    print("- Follows PyTorch Lightning best practices")


if __name__ == "__main__":
    print("NablAFx Callback System Demo")
    print("=" * 40)

    # Show YAML config example
    demo_yaml_config()

    # Uncomment to run actual training
    # main()
