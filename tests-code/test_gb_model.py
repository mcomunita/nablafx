import torch

from nablafx.processors import Gain, DCOffset, ParametricEQ, Peaking, TanhNonlinearity
from nablafx.interfaces import Processor, Controller
from nablafx.models import GreyBoxModel

blocks = [
    Gain(44100.0, -6, 6, "static"),
    Gain(44100.0, -6, 6, "static-cond"),
    Gain(44100.0, -6, 6, "dynamic"),
    Gain(44100.0, -6, 6, "dynamic-cond"),
    DCOffset(44100.0, -2, 2, "static"),
    DCOffset(44100.0, -2, 2, "static-cond"),
    DCOffset(44100.0, -2, 2, "dynamic"),
    DCOffset(44100.0, -2, 2, "dynamic-cond"),
    ParametricEQ(44100.0, control_type="static"),
    ParametricEQ(44100.0, control_type="static-cond"),
    ParametricEQ(44100.0, control_type="dynamic"),
    ParametricEQ(44100.0, control_type="dynamic-cond"),
    Peaking(44100.0, control_type="static"),
    Peaking(44100.0, control_type="static-cond"),
    Peaking(44100.0, control_type="dynamic"),
    Peaking(44100.0, control_type="dynamic-cond"),
    TanhNonlinearity(44100.0),
]

model = GreyBoxModel(
    blocks,
    num_controls=2,
)

x = torch.randn(4, 1, 1024)
controls = torch.randn(4, 2)

y = model(x, controls)
print("x: ", x.shape)
print("y: ", y.shape)

blocks = [
    Gain(44100.0, -6, 6, "static"),
    Gain(44100.0, -6, 6, "dynamic"),
    DCOffset(44100.0, -2, 2, "static"),
    DCOffset(44100.0, -2, 2, "dynamic"),
    ParametricEQ(44100.0, control_type="static"),
    ParametricEQ(44100.0, control_type="dynamic"),
    Peaking(44100.0, control_type="static"),
    Peaking(44100.0, control_type="dynamic"),
    TanhNonlinearity(44100.0),
]

model = GreyBoxModel(
    blocks,
    num_controls=0,
)

x = torch.randn(4, 1, 1024)

y = model(x, None)
print("x: ", x.shape)
print("y: ", y.shape)
