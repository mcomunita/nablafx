import torch
from calflops import calculate_flops
from nablafx.models import GreyBoxModel
from nablafx.processors import *

SECONDS = 1
NUM_CONTROLS = 4

processors = [
    ParametricEQ(
        sample_rate=48000,
        control_type="static-cond",
    ),
    Gain(
        sample_rate=48000,
        control_type="static-cond",
    ),
    DCOffset(
        sample_rate=48000,
        control_type="dynamic-cond",
    ),
    StaticMLPNonlinearity(
        sample_rate=48000,
        hidden_dim=32,
        num_layers=3,
    ),
    Gain(
        sample_rate=48000,
        control_type="static-cond",
    ),
    ParametricEQ(
        sample_rate=48000,
        control_type="static-cond",
    ),
    Lowpass(sample_rate=48000, control_type="static"),
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

x = torch.rand(1, 1, 48000 * SECONDS)
p = torch.rand(1, model.num_controls) if model.num_controls > 0 else None

flops, macs, params = calculate_flops(
    model=model, input_shape=None, kwargs={"x": x, "controls": p} if p is not None else {"x": x}, output_as_string=True, output_precision=4
)
