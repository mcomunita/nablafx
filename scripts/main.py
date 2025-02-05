import os
import torch
import wandb

# from pytorch_lightning.cli import LightningCLI
# from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning import seed_everything
from lightning.pytorch.cli import LightningCLI
from lightning import seed_everything

seed_everything(seed=42, workers=True)
torch.set_float32_matmul_precision("high")
torch.use_deterministic_algorithms(False, warn_only=True)
torch.set_num_threads(1)  # number of CPUs
os.environ["WANDB__SERVICE_WAIT"] = "300"


def cli_main():
    _ = LightningCLI(
        seed_everything_default=False,
        trainer_defaults={
            "accelerator": "gpu",
            # "strategy": DDPStrategy(find_unused_parameters=False),  # GCN = True, any other model = False
            "devices": -1,
            "num_sanity_val_steps": 2,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 100,
            # "max_steps": 150000,
            "sync_batchnorm": True,
            "enable_model_summary": True,
            # "gradient_clip_val": 4.0,
            # "gradient_clip_algorithm": "norm",
            "enable_checkpointing": True,
            "deterministic": None,
            "benchmark": True,
        },
        save_config_kwargs={
            # "config_filename": "config.yaml",
            "overwrite": False,
        },
    )


if __name__ == "__main__":
    cli_main()
    wandb.finish()
