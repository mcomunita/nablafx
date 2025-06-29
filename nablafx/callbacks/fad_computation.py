"""
FAD Computation Callback

Handles computation and logging of Fréchet Audio Distance (FAD) scores.
Extracted from system.py to provide modular, configurable FAD computation.
"""

import os
import torch
import wandb
from typing import Any, Dict, List, Optional
import lightning as pl
from lightning.pytorch.callbacks import Callback
from frechet_audio_distance import FrechetAudioDistance


class FADComputationCallback(Callback):
    """
    Callback for computing and logging Fréchet Audio Distance (FAD) scores.

    This callback handles the computation of FAD scores using different pre-trained
    models (VGGish, PANN, CLAP, AFX-Rep) to evaluate the quality of generated audio.

    Args:
        compute_on_train_end: Whether to compute FAD at training end (default: True)
        compute_on_test_end: Whether to compute FAD at test end (default: True)
        compute_every_n_epochs: Compute FAD every N epochs during validation (default: None)
        models: List of FAD models to use (default: ["vggish", "pann", "clap", "afx-rep"])
        checkpoint_dir: Directory containing FAD model checkpoints (default: "checkpoints_fad")
        model_configs: Custom configurations for each FAD model (optional)
    """

    def __init__(
        self,
        compute_on_train_end: bool = True,
        compute_on_test_end: bool = True,
        compute_every_n_epochs: Optional[int] = None,
        models: List[str] = None,
        checkpoint_dir: str = "checkpoints_fad",
        model_configs: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__()
        self.compute_on_train_end = compute_on_train_end
        self.compute_on_test_end = compute_on_test_end
        self.compute_every_n_epochs = compute_every_n_epochs
        self.models = models or ["vggish", "pann", "clap", "afx-rep"]
        self.checkpoint_dir = checkpoint_dir
        self.model_configs = model_configs or {}

        # Initialize FAD models lazily
        self._fad_models = {}

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Compute FAD at training end."""
        if self.compute_on_train_end:
            self._compute_and_log_fad(trainer, pl_module, "train_end")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Compute FAD at test end."""
        if self.compute_on_test_end:
            self._compute_and_log_fad(trainer, pl_module, "test")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Compute FAD every N validation epochs."""
        if self.compute_every_n_epochs is not None and trainer.current_epoch % self.compute_every_n_epochs == 0:
            self._compute_and_log_fad(trainer, pl_module, "val")

    def _get_fad_model(self, model_name: str) -> FrechetAudioDistance:
        """Get or create FAD model instance."""
        if model_name not in self._fad_models:
            # Get checkpoint directory
            parent_dir = os.path.abspath(os.getcwd())
            ckpt_dir = os.path.join(parent_dir, self.checkpoint_dir)
            model_ckpt_dir = os.path.join(ckpt_dir, model_name)

            # Get model-specific configuration
            config = self.model_configs.get(model_name, {})

            # Default configurations for each model
            default_configs = {
                "vggish": {
                    "model_name": "vggish",
                    "sample_rate": 16000,
                    "use_pca": False,
                    "use_activation": False,
                    "verbose": False,
                },
                "pann": {
                    "model_name": "pann",
                    "sample_rate": 32000,
                    "verbose": False,
                },
                "clap": {
                    "model_name": "clap",
                    "submodel_name": "630k-audioset",
                    "sample_rate": 48000,
                    "verbose": False,
                    "enable_fusion": False,
                },
                "afx-rep": {
                    "model_name": "afx-rep",
                    "sample_rate": 48000,
                    "verbose": False,
                },
            }

            # Merge default config with user config
            final_config = default_configs.get(model_name, {})
            final_config.update(config)

            # Create FAD model
            self._fad_models[model_name] = FrechetAudioDistance(model_ckpt_dir, **final_config)

        return self._fad_models[model_name]

    def _compute_and_log_fad(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        phase: str,
    ) -> None:
        """Compute and log FAD scores."""
        if not hasattr(trainer.logger, "experiment"):
            return  # Skip if no wandb logger

        try:
            print(f"\nComputing and logging FAD for {phase}...")

            # Get audio directories
            run_dir = trainer.logger.experiment.dir
            pred_dir = os.path.join(run_dir, f"media/audio/audio/{phase}/pred")
            target_dir = os.path.join(run_dir, f"media/audio/audio/{phase}/target")

            # Check if directories exist
            if not os.path.exists(pred_dir) or not os.path.exists(target_dir):
                print(f"Warning: Audio directories not found for FAD computation in {phase}")
                return

            print(f"Computing FAD for {run_dir}...")

            # Compute FAD scores for each model
            fad_scores = {}
            for model_name in self.models:
                try:
                    print(f"Computing FAD score using {model_name}...")
                    fad_model = self._get_fad_model(model_name)

                    fad_score = fad_model.score(target_dir, pred_dir)
                    fad_scores[model_name] = fad_score

                    print(f"FAD score ({model_name}): {fad_score}")

                except Exception as e:
                    print(f"Warning: Failed to compute FAD with {model_name}: {e}")
                    continue

            # Log FAD scores
            if fad_scores:
                log_dict = {f"metrics/{phase}/fad-{model_name}": score for model_name, score in fad_scores.items()}

                trainer.logger.experiment.log(log_dict, step=trainer.global_step)

                print(f"Successfully logged {len(fad_scores)} FAD scores for {phase}")
            else:
                print(f"Warning: No FAD scores computed for {phase}")

        except Exception as e:
            print(f"Warning: Failed to compute FAD for {phase}: {e}")
            # Don't crash training if FAD computation fails
