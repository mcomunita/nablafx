import torch

from nnlafx.modules import MLP, GCNCondBlock, TFiLM, TinyTFiLM, TVFiLMCond, causal_crop, center_crop


"""
GCN (Conditional):

- parametric or non-parametric
- conditioning with FiLM, TFilm, TTFiLM or TVFiLM
- optional direct path
"""


class GCN(torch.nn.Module):
    """
    cond_type:  if None -> basic GCN, no modulation, no parametric control
                if film -> cond_dim = conditioning MLP output size
                if tfilm -> cond_dim = number of control parameters
                if ttfilm -> cond_dim = number of control parameters
                if tvfilm -> cond_dim = TVFiLMCond out_nfeatures
    """

    def __init__(
        self,
        num_inputs=1,
        num_outputs=1,
        num_controls=0,
        num_blocks=10,
        kernel_size=3,
        dilation_growth=2,
        channel_growth=1,
        channel_width=32,
        stack_size=10,
        groups=1,
        bias=False,
        causal=False,
        batchnorm=False,
        residual=False,
        direct_path=False,
        cond_type=None,
        cond_block_size=128,
        cond_num_layers=1,
    ):
        super(GCN, self).__init__()
        assert cond_type in [None, "film", "tfilm", "ttfilm", "tvfilm"]
        if cond_type == "film":
            assert num_controls > 0  # film conditioning can be used only for parametric models
        if cond_type in ["tfilm", "ttfilm", "tvfilm"]:
            assert num_controls >= 0  # tfilm, ttfilm and tvfilm conditioning can be used for parametric and non-parametric models

        # PARAMS
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_controls = num_controls
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.channel_growth = channel_growth
        self.channel_width = channel_width
        self.stack_size = stack_size
        self.groups = groups
        self.bias = bias
        self.causal = causal
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
        self.rf = self.compute_receptive_field()

        # CONDITIONING
        if cond_type == "film":  # conditioning MLP
            self.cond_nn = MLP(input_dim=num_controls, output_dim=self.cond_dim)
        elif cond_type == "tfilm":  # conditioning is passed to each block during forward
            pass
        elif cond_type == "tvfilm":  # conditioning LSTM
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

        # BLOCKS
        self.blocks = torch.nn.ModuleList()
        for n in range(num_blocks):
            in_ch = num_inputs if n == 0 else out_ch

            out_ch = channel_width

            dilation = dilation_growth ** (n % stack_size)
            self.blocks.append(
                GCNCondBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    causal=causal,
                    batchnorm=batchnorm,
                    residual=residual,
                    kernel_size=kernel_size,
                    padding=0,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    cond_type=cond_type,
                    cond_dim=self.cond_dim,
                    cond_block_size=cond_block_size,
                    cond_num_layers=cond_num_layers,
                )
            )

        # MIX
        self.mix = torch.nn.Conv1d(channel_width * num_blocks, num_outputs, kernel_size=1, bias=bias)

    def forward(self, x, p=None):
        # x = input : (batch, channels, seq)
        # p = params : (batch, params)

        z = []  # outputs from each block

        # PADDING
        if self.causal:
            x = torch.nn.functional.pad(x, (self.rf - 1, 0))
        else:
            x = torch.nn.functional.pad(x, ((self.rf - 1) // 2, (self.rf - 1) // 2))

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

        # BLOCKS
        x_proc = x

        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            x_proc, zn = block(x_proc, cond)
            z.append(zn)

        # crop outputs to have same length
        for idx, zn in enumerate(z):
            if self.causal:
                z[idx] = causal_crop(zn, z[-1].shape[-1])
            else:
                z[idx] = center_crop(zn, z[-1].shape[-1])

        # concatenate outputs
        z = torch.cat(z, dim=1)

        # MIX
        y_proc = self.mix(z)

        # OUTPUT
        if self.direct_path:
            if self.causal:
                y_direct = causal_crop(y_direct, y_proc.shape[-1])
            else:
                y_direct = center_crop(y_direct, y_proc.shape[-1])

            out = y_direct + y_proc
        else:
            out = y_proc

        return out

    def compute_receptive_field(self):
        """
        compute the receptive field in samples.
        """
        rf = self.kernel_size
        for n in range(1, self.num_blocks):
            dilation = self.dilation_growth ** (n % self.stack_size)
            rf = rf + ((self.kernel_size - 1) * dilation)
        return rf

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
