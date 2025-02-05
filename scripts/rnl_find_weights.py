from rational.utils.find_init_weights import find_weights
import torch.nn.functional as F

"""
Find the initial weight for the Rational Activation Function
"""

find_weights(F.tanh)
