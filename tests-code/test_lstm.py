import torch

from itertools import product
from nablafx.lstm import LSTM

residual = [True, False]
direct_path = [True, False]
cond_type = [None, "fixed", "tvcond"]

i = 0
for r,dp,ct in product(residual, direct_path, cond_type):
    model = LSTM(
        num_inputs=1
        num_outputs=1
        num_controls=0
        hidden_size=32
        num_layers=1
        residual=false
        direct_path=false
        cond_type=null # null, fixed, tvcond
        cond_block_size=128 # block size for tvcond
        cond_num_layers=1 # number of lstm layers for tvcond
    )

    print(f"Test model {i+1} - r:{r}, dp:{dp}, ct:{ct}")
    print("Trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.rand(1, 1, 1050)
    p = torch.rand(1, model.nparams) if model.nparams > 0 else None
    y = model(x, p)

    assert x.shape == y.shape

    i+=1

    