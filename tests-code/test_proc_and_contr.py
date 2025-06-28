import torch

from nablafx.processors import Gain, DCOffset, ParametricEQ, Peaking, TanhNonlinearity
from nablafx.interfaces import Processor, Controller

blocks_1 = [
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

blocks_2 = [
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

processor_1 = Processor(processors=blocks_1)
controller_1 = Controller(processor_1, 0)

processor_2 = Processor(processors=blocks_2)
controller_2 = Controller(processor_2, 2)

x = torch.randn(4, 1, 1024)
controls = torch.randn(4, 2)

control_params = controller_1(x)
for c in control_params:
    print("block control params:", c.shape)

y = processor_1(x, control_params)
print("x: ", x.shape)
print("y: ", y.shape)

control_params = controller_2(x, controls)
for c in control_params:
    print("block control params:", c.shape)

y = processor_2(x, control_params)
print("x: ", x.shape)
print("y: ", y.shape)
