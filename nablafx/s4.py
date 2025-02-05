import os
import torch
from einops import rearrange
from torch import Tensor

import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from nablafx.modules import MLP, TCNCondBlock, TFiLM, TinyTFiLM, TVFiLMCond
from nablafx.modules import S4CondBlock


class S4(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int = 1,
        num_outputs: int = 1,
        num_controls: int = 0,
        num_blocks: int = 4,
        channel_width: int = 32,
        s4_state_dim: int = 4,
        batchnorm: bool = False,
        residual: bool = False,
        direct_path: bool = False,
        cond_type: str = None,
        cond_block_size: int = 128,
        cond_num_layers: int = 1,
        act_type: str = "tanh",
        s4_learning_rate: float = 0.0005,
    ):
        super().__init__()
        assert num_inputs == 1, f"implemented only for 1 input channels"
        assert num_outputs == 1, f"implemented only for 1 output channels"
        assert cond_type in [None, "film", "tfilm", "ttfilm", "tvfilm"]
        assert act_type in ["tanh", "prelu", "rational"]
        if cond_type == "film":
            assert num_controls > 0  # film conditioning can be used only for parametric models
        if cond_type in ["tfilm", "ttfilm", "tvfilm"]:
            # tfilm and tvfilm conditioning can be used for parametric and non-parametric models
            assert num_controls >= 0
        assert channel_width >= 1, f"The inner audio channel is expected to be one or greater, but got {channel_width}."
        assert s4_state_dim >= 1, f"The S4 hidden size is expected to be one or greater, but got {s4_state_dim}."
        assert num_blocks >= 0, f"The model depth is expected to be zero or greater, but got {num_blocks}."

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_controls = num_controls
        self.num_blocks = num_blocks
        self.channel_width = channel_width
        self.s4_state_dim = s4_state_dim
        self.batchnorm = batchnorm
        self.residual = residual
        self.direct_path = direct_path
        self.cond_type = cond_type
        if cond_type is None:
            self.cond_dim = 0
        elif cond_type == "film":
            self.cond_dim = 32
        elif cond_type == "tfilm":
            self.cond_dim = num_controls
        elif cond_type == "ttfilm":
            self.cond_dim = num_controls
        elif cond_type == "tvfilm":
            self.cond_dim = 32
        self.cond_block_size = cond_block_size
        self.cond_num_layers = cond_num_layers
        self.act_type = act_type

        if cond_type == "film":  # conditioning MLP
            self.cond_nn = MLP(input_dim=num_controls, output_dim=self.cond_dim)
        elif cond_type == "tfilm":  # conditioning is passed to each block during forward
            pass
        elif cond_type == "tvfilm":  # conditioning Pool+LSTM
            self.cond_nn = TVFiLMCond(
                input_dim=num_inputs,
                output_dim=self.cond_dim,
                cond_dim=num_controls,
                block_size=cond_block_size,
                num_layers=cond_num_layers,
            )

        # DIRECT PATH
        if direct_path:
            self.direct_gain = torch.nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)

        # INPUT
        self.expand = torch.nn.Linear(num_inputs, channel_width)

        # BLOCKS
        self.blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                S4CondBlock(
                    channel_width=channel_width,
                    batchnorm=batchnorm,
                    residual=residual,
                    s4_state_dim=s4_state_dim,
                    s4_learning_rate=s4_learning_rate,
                    cond_type=cond_type,
                    cond_dim=self.cond_dim,
                    cond_block_size=cond_block_size,
                    cond_num_layers=cond_num_layers,
                    act_type=act_type,
                )
            )

        # OUTPUT
        self.contract = torch.nn.Linear(channel_width, num_outputs)

    def _pass_blocks(self, x: Tensor, cond: Tensor = None) -> Tensor:
        x = rearrange(x, "B L -> B L 1")

        x = self.expand(x)

        x = rearrange(x, "B L H -> B H L")
        for block in self.blocks:
            x = block(x, cond)
        x = rearrange(x, "B H L -> B L H")

        x = self.contract(x)

        x = rearrange(x, "B H 1 -> B H")

        return x

    def forward(self, x: Tensor, p: Tensor = None) -> Tensor:
        # x = input : (batch, channels, seq)
        # p = params : (batch, params)
        bs, chs, seq_len = x.size()
        assert chs == 1, "The input tensor is expected to have one channel."

        # CONDITIONING
        if self.cond_type is None:
            cond = None
        elif self.cond_type == "film":
            cond = self.cond_nn(p)
        elif self.cond_type == "tfilm":
            cond = p
        elif self.cond_type == "ttfilm":
            cond = p
        elif self.cond_type == "tvfilm":
            cond = self.cond_nn(x, p)

        # DIRECT PATH
        if self.direct_path:
            y_direct = self.direct_gain(x)

        x = x.view(bs, seq_len)
        y_proc = self._pass_blocks(x, cond)
        y_proc = y_proc.view(bs, 1, seq_len)  # add channel dimension

        # OUTPUT
        if self.direct_path:
            out = torch.tanh(y_direct + y_proc)
        else:
            out = torch.tanh(y_proc)

        return out

    def reset_states(self):
        """
        reset state for all TFiLM, TinyTFiLM and TVFilMCond layers
        """
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.reset_state()
            elif isinstance(layer, TinyTFiLM):
                layer.reset_state()
            elif isinstance(layer, TVFiLMCond):
                layer.reset_state()
