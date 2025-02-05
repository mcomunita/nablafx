import torch
from calflops import calculate_flops
from nablafx.lstm import LSTM

SECONDS = 1
NUM_CONTROLS = 0

model = LSTM(
    num_inputs=1,
    num_outputs=1,
    num_controls=NUM_CONTROLS,
    hidden_size=1,
    num_layers=1,
    residual=False,
    direct_path=False,
    cond_type=None,  # null, fixed, tvcond
    cond_block_size=128,  # block size for tvcond
    cond_num_layers=1,  # number of lstm layers for tvcond
)

x = torch.rand(1, 1, 48000 * SECONDS)
p = torch.rand(1, model.num_controls) if model.num_controls > 0 else None

flops, macs, params = calculate_flops(
    model=model, input_shape=None, kwargs={"x": x, "p": p} if p is not None else {"x": x}, output_as_string=True, output_precision=4
)
