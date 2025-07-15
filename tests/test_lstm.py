import torch

from itertools import product
from nablafx.lstm import LSTM

residual = [True, False]
direct_path = [True, False]
cond_type = [None, "fixed", "tvcond"]

i = 0
for r,dp,ct in product(residual, direct_path, cond_type):
    # Set num_controls based on cond_type requirements
    if ct == "fixed":
        num_controls = 2  # fixed conditioning requires num_controls > 0
    else:
        num_controls = 0  # None and tvcond can work with 0 controls
        
    model = LSTM(
        num_inputs=1,
        num_outputs=1,
        num_controls=num_controls,
        hidden_size=32,
        num_layers=1,
        residual=r,
        direct_path=dp,
        cond_type=ct,
        cond_block_size=128,  # block size for tvcond
        cond_num_layers=1  # number of lstm layers for tvcond
    )

    print(f"Test model {i+1} - r:{r}, dp:{dp}, ct:{ct}")
    print("Trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.rand(1, 1, 1050)
    p = torch.rand(1, model.num_controls) if model.num_controls > 0 else None
    y = model(x, p)

    try:
        assert x.shape == y.shape, f"Shape mismatch: x={x.shape}, y={y.shape} for config r:{r}, dp:{dp}, ct:{ct}"
        print(f"✓ Test model {i+1} passed - shapes match: {x.shape}")
    except AssertionError as e:
        print(f"✗ Test model {i+1} failed: {e}")
        raise

    i+=1

    