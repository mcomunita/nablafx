import torch
from nablafx.models import GreyBoxModel
from nablafx.processors import *

NUM_CONTROLS = 0

processors = [
    ParametricEQ(
        sample_rate=48000,
        block_size=128,
        control_type="static",
    ),
    Gain(
        sample_rate=48000,
        control_type="dynamic",
    ),
    ParametricEQ(
        sample_rate=48000,
        block_size=128,
        control_type="static",
    ),
    Gain(
        sample_rate=48000,
        control_type="static",
    ),
]

model = GreyBoxModel(
    processors=processors,
    num_controls=NUM_CONTROLS,
    stat_control_params_initial=0.0,
    stat_cond_num_layers=3,
    stat_cond_hidden_dim=16,
    dyn_block_size=128,
    dyn_num_layers=1,
    dyn_cond_block_size=128,
    dyn_cond_num_layers=1,
)

print(f"Test model")
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

x = torch.rand(1, 1, 48000)
p = torch.rand(1, model.num_controls) if model.num_controls > 0 else None
y = model(x, p)

assert x.shape == y.shape
