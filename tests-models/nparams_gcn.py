import torch
from nablafx.gcn import GCN

model = GCN(
    num_inputs=1,
    num_outputs=1,
    num_controls=0,
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
    cond_type="tfilm",  # null, film, tfilm, tvfilm
    cond_block_size=128,  # block size for tfilm or tvfilm
    cond_num_layers=1,  # number of lstm layers for tfilm or tvfilm
)

print(f"Test model")
print(model.rf)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

x = torch.rand(1, 1, model.rf)
p = torch.rand(1, model.num_controls) if model.num_controls > 0 else None
y = model(x, p)

assert x.shape == y.shape
