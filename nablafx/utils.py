import torch

from torch import Tensor
from typing import Literal, get_args
from einops import rearrange


class Rearrange(torch.nn.Module):
    """Tensor reshape layer.

    This layer wraps the `einops.rearrange` method as an neural network layer
    to streamline tensor reshape operations.
    Useful for `nn.Sequential` layers.

    The constructor of the layer follows the same signature as the `einops.rearrange`
    except the first argument, which is the input tensor.
    See `einops.rearrange` method for help.

    ```python
    x = torch.rand(3, 6)
    layer = Rearrange('b (c d) -> b d c', c=2)
    x = layer(x)
    x.size()
    >>> torch.Size([3, 3, 2])
    ```
    """

    pattern: str
    axes_lengths: dict[str, int]

    def __init__(self, pattern: str, **axes_lengths: int) -> None:
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, self.pattern, **self.axes_lengths)


class PTanh(torch.nn.Module):
    def __init__(self, a: float = 1.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a))

    def forward(self, x: Tensor) -> Tensor:
        return self.a * torch.tanh(x)
