import torch
from nablafx.lstm import LSTM

model = LSTM(
    num_inputs=1,
    num_outputs=1,
    num_controls=0,
    hidden_size=96,
    num_layers=1,
    residual=False,
    direct_path=False,
    cond_type=None,  # null, fixed, tvcond
    cond_block_size=128,  # block size for tvcond
    cond_num_layers=1,  # number of lstm layers for tvcond
)

print("Trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

x = torch.rand(1, 1, 48000)
p = torch.rand(1, model.num_controls) if model.num_controls > 0 else None
y = model(x, p)

assert x.shape == y.shape
