# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About NablAFx

NablAFx is a PyTorch-based framework for differentiable black-box and gray-box modeling of audio effects. It's research software that models nonlinear audio effects using neural networks, with support for various architectures including LSTMs, TCNs, GCNs, and S4 models.

## Development Environment Setup

The project requires Python 3.9.7 and uses both pip and conda for dependency management.

**Conda setup (recommended):**
```bash
conda env create -f environment.yml
conda env update -f environment.yml
conda activate nablafx
# Move weights/rationals_config.json to ~/.conda/envs/nablafx/lib/python3.9/site-packages/rational/rationals_config.json
```

**Pip setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_temp-for-rnl.txt
pip install --upgrade -r requirements.txt
# Move weights/rationals_config.json to .venv/lib/python3.9/site-packages/rational/rationals_config.json
```

## Core Commands

**Training:**
- Black-box models: `bash scripts/train_bb.sh`
- Gray-box models: `bash scripts/train_gb.sh`
- Example: `CUDA_VISIBLE_DEVICES=0 python scripts/main.py fit -c cfg/data/data-param_multidrive-ffuzz_trainval.yaml -c cfg/model/s4-param/model_bb-param_s4-tvf-b8-s32-c16.yaml -c cfg/trainer/trainer_bb.yaml`

**Testing:**
- `bash scripts/test.sh` (modify paths for specific models)
- Example: `CUDA_VISIBLE_DEVICES=0 python scripts/main.py test --config [config.yaml] --ckpt_path [checkpoint.ckpt]`

**Pretraining processors:**
- `python scripts/pretrain.py` - for differentiable processors like StaticMLPNonlinearity or StaticFIRFilter

**Performance analysis:**
- Various scripts in `tests/` folder measure parameters, FLOPs, MACs, and real-time factors
- Example: `python tests/flops-macs-params_tcn.py`

## Architecture Overview

The framework is organized around several key concepts:

**Core Models:**
- `BlackBoxModel` (`nablafx/models.py`): Single neural network processor
- `GreyBoxModel` (`nablafx/models.py`): Chain of differentiable processors and controllers

**Neural Network Architectures:**
- LSTM (`nablafx/lstm.py`): Recurrent models for temporal modeling
- TCN (`nablafx/tcn.py`): Temporal Convolutional Networks
- GCN (`nablafx/gcn.py`): Gated Convolutional Networks  
- S4 (`nablafx/s4.py`): State Space Models

**Key Components:**
- `system.py`: Lightning modules (BaseSystem) with training/validation logic
- `data.py`: Dataset classes for ToneTwist AFx Dataset
- `processors.py`: Differentiable DSP blocks (filters, nonlinearities)
- `controllers.py`: Parameter conditioning mechanisms
- `loss.py`: Time and frequency domain loss functions
- `dsp.py`: Core DSP utilities and functions

**Configuration System:**
- Uses Lightning CLI with YAML configs in `cfg/` directory
- Separate configs for data (`cfg/data/`), models (`cfg/model/`), and training (`cfg/trainer/`)
- Model configs organized by architecture type (tcn/, lstm/, s4/, gcn/, gb/)

## Data Requirements

- Setup data directory: `mkdir data && cd data && ln -s /path/to/TONETWIST-AFX-DATASET/`
- Create logs directory: `mkdir logs`
- FAD checkpoints: `mkdir checkpoints_fad` (auto-downloaded on first use)

## Logging and Monitoring

- Uses Weights & Biases for experiment tracking
- Run `wandb login` before training
- Logs saved to `logs/` directory with hierarchical organization

## Testing Framework

Performance testing scripts in `tests/` directory measure:
- Model parameters, FLOPs, MACs
- CPU speed and real-time factors  
- Individual component tests (test_*.py files)

Use `bash tests/speed_run_with_priority.sh` for batch performance testing.

## Important Notes

- This is research software - no backward compatibility required
- Uses rational activations library with custom config placement
- GPU training typically uses single device (CUDA_VISIBLE_DEVICES=0)
- Models support both parametric and non-parametric conditioning
- Supports both black-box (end-to-end neural) and gray-box (differentiable DSP + neural) approaches