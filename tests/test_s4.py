import torch
from nablafx.s4 import S4

model = S4(
    num_controls=2,
    channel_width=32,
    s4_state_dim=4,
    s4_learning_rate=0.005,
    num_blocks=4,
    batchnorm=True,
    residual=True,
    act_type="prelu",
)

# test model
bs = 8
x = torch.rand(bs, 1, 65536)
p = torch.rand(bs, 2)

y = model(x, p)
print(y.shape)


# CUDA_VISIBLE_DEVICES=1 python scripts/main.py fit \
# -c cfg/trainer/trainer_bb_s4.yaml \
# -c cfg/data/data-param_multidrive-ffuzz.yaml \
# -c cfg/model/bb_s4-L-4-32.yaml
