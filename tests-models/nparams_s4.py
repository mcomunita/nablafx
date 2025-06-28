import torch

from itertools import product
from nablafx.s4 import S4

model = S4(
    num_inputs=1,
    num_outputs=1,
    num_controls=0,
    num_blocks=8,
    s4_state_dim=32,
    channel_width=16,
    batchnorm=False,
    residual=True,
    direct_path=False,
    cond_type="tfilm",
    cond_block_size=128,
    cond_num_layers=1,
    act_type="tanh",
    s4_learning_rate=0.01,
)

print("Trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

x = torch.rand(1, 1, 48000)
p = torch.rand(1, model.num_controls) if model.num_controls > 0 else None
y = model(x, p)

assert x.shape == y.shape
