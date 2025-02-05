import torch

from nablafx.dist import Distortion
from nablafx.dsp import Gain, TanhNonlinearity, ParametricEQ, StaticMLPNonlinearity

sample_rate = 48000

prefilter = ParametricEQ(sample_rate, -32.0, 32.0)
nonlinearity = StaticMLPNonlinearity(sample_rate)
postfilter = ParametricEQ(sample_rate, -32.0, 32.0)

model = Distortion(prefilter, nonlinearity, postfilter)

bs = 1
chs = 1
seq_len = 262144

x = torch.randn(bs, chs, seq_len)
prefilter_params = torch.rand(bs, prefilter.num_control_params)
nonlinearity_params = torch.rand(bs, nonlinearity.num_control_params)
postfilter_params = torch.rand(bs, postfilter.num_control_params)

if prefilter_params is not None:
    params = prefilter_params
if nonlinearity_params is not None:
    params = torch.cat([params, nonlinearity_params], dim=1)
if postfilter_params is not None:
    params = torch.cat([params, postfilter_params], dim=1)

y = model(x, params, train=False)
