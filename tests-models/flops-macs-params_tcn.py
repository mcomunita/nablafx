import torch
from calflops import calculate_flops
from nablafx.tcn import TCN

SECONDS = 1
NUM_CONTROLS = 4

model = TCN(
    num_inputs=1,
    num_outputs=1,
    num_controls=NUM_CONTROLS,
    num_blocks=10,
    kernel_size=5,
    dilation_growth=3,
    channel_width=16,
    channel_growth=1,
    stack_size=12,
    groups=1,
    bias=True,
    causal=True,
    batchnorm=False,  # active only when cond_type = None
    residual=True,
    direct_path=False,
    cond_type="tvfilm",  # null, film, tfilm, tvfilm
    cond_block_size=128,  # block size for tfilm or tvfilm
    cond_num_layers=1,  # number of lstm layers for tfilm or tvfilm
)

x = torch.rand(1, 1, 48000 * SECONDS)
p = torch.rand(1, model.num_controls) if model.num_controls > 0 else None

flops, macs, params = calculate_flops(
    model=model, input_shape=None, kwargs={"x": x, "p": p} if p is not None else {"x": x}, output_as_string=True, output_precision=4
)
