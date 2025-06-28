import torch
from nablafx.s4 import S4ConditionalModel

model = S4ConditionalModel(
    num_control_params=2,
    take_side_chain=False,
    inner_audio_channel=32,
    s4_hidden_size=4,
    s4_learning_rate=0.005,
    model_depth=4,
    film_take_batchnorm=True,
    take_residual_connection=True,
    convert_to_decibels=False,
    take_tanh=True,
    activation="PTanh",
    take_parametered_tanh=True,
    convert_to_amplitude=False,
)

# test model
bs = 8
x = torch.rand(bs, 1, 65536)
p = torch.rand(bs, 2)

y = model(x, p)
print(y.shape)


CUDA_VISIBLE_DEVICES=1 python scripts/main.py fit \
-c cfg/trainer/trainer_bb_s4.yaml \
-c cfg/data/data-param_multidrive-ffuzz.yaml \
-c cfg/model/bb_s4-L-4-32.yaml \
