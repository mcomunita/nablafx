import torch

from itertools import product
from nablafx.tcn import TCN

bias = [True, False]
causal = [True, False]
batchnorm = [True, False]
residual = [True, False]
direct_path = [True, False]
cond_type = [None, "film", "tfilm", "tvfilm"]

i = 0
for b, c, bn, r, dp, ct in product(bias, causal, batchnorm, residual, direct_path, cond_type):
    model = TCN(
        ninputs=1,
        noutputs=1,
        nparams=4,
        nblocks=10,
        kernel_size=3,
        dilation_growth=2,
        channel_growth=1,
        channel_width=32,
        stack_size=10,
        groups=1,
        bias=b,
        causal=c,
        batchnorm=bn,
        residual=r,
        direct_path=dp,
        cond_type=ct,
        cond_block_size=128,
        cond_nlayers=1,
    )

    print(f"Test model {i+1} - b:{b}, c:{c}, bn:{bn}, r:{r}, dp:{dp}, ct:{ct}")
    print(model.rf)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.rand(1, 1, model.rf)
    p = torch.rand(1, model.nparams) if model.nparams > 0 else None
    y = model(x, p)

    assert x.shape == y.shape

    i += 1
