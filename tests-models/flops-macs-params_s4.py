import torch
from calflops import calculate_flops
from nablafx.s4 import S4

SECONDS = 1
NUM_CONTROLS = 4

model = S4(
    num_inputs=1,
    num_outputs=1,
    num_controls=NUM_CONTROLS,
    num_blocks=4,
    s4_state_dim=4,
    channel_width=16,
    batchnorm=False,
    residual=True,
    direct_path=False,
    cond_type="film",
    cond_block_size=128,
    cond_num_layers=1,
    act_type="tanh",
    s4_learning_rate=0.01,
)

x = torch.rand(1, 1, 48000 * SECONDS)
p = torch.rand(1, model.num_controls) if model.num_controls > 0 else None

flops, macs, params = calculate_flops(
    model=model, input_shape=None, kwargs={"x": x, "p": p} if p is not None else {"x": x}, output_as_string=True, output_precision=4
)
