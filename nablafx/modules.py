import math
import torch
from einops import rearrange, repeat
from torch import Tensor
from rational.torch import Rational


# -----------------------------------------------------------------------------
# Crop functions for residual connections
# -----------------------------------------------------------------------------


def center_crop(x, length: int):
    if x.shape[-1] > length:
        start = (x.shape[-1] - length) // 2
        stop = start + length
        return x[..., start:stop]
    else:
        return x


def causal_crop(x, length: int):
    if x.shape[-1] > length:
        stop = x.shape[-1] - 1
        start = stop - length
        return x[..., start:stop]
    else:
        return x


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------


class MLP(torch.nn.Module):
    """
    MLP used in conditioning layers

    num_layers
    input_dim: input dimensions
    hidden_dim: hidden dimensions
    output_dim: output dimensions
    """

    def __init__(self, input_dim, output_dim, num_layers=3, hidden_dim=32, activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation)

        self.mlp = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.mlp(x)


# -----------------------------------------------------------------------------
# FiLM layer
# -----------------------------------------------------------------------------


class FiLM(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond, modulates x.

    nfeatures: number of features (i.e., convolution channels)
    cond_dim: number of conditioning features
    """

    def __init__(self, nfeatures, cond_dim):
        super(FiLM, self).__init__()
        self.nfeatures = nfeatures
        self.cond_dim = cond_dim

        self.bn = torch.nn.BatchNorm1d(nfeatures, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, nfeatures * 2)

    def forward(self, x, cond):
        # x = [batch, channels, length]
        # cond = [batch, cond_dim]
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)  # [batch, nfeatures, 1]
        b = b.unsqueeze(-1)  # [batch, nfeatures, 1]
        x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # apply conditional affine
        return x


# -----------------------------------------------------------------------------
# Temporal FiLM layer
# -----------------------------------------------------------------------------


class TFiLM(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond,
    modulates x over time.

    nfeatures: number of features (i.e., convolution channels)
    cond_dim: number of control parameters, if cond_dim = 0, modulation is only a function of features
    block_size: size of blocks to modulate
    num_layers: number of LSTM layers
    """

    def __init__(self, nfeatures, cond_dim, block_size, num_layers):
        super(TFiLM, self).__init__()
        self.nfeatures = nfeatures
        self.cond_dim = cond_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.hidden_state = None  # (hidden_state, cell_state)

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(kernel_size=block_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.lstm = torch.nn.LSTM(
            input_size=nfeatures + cond_dim, hidden_size=nfeatures * 2, num_layers=num_layers, batch_first=False, bidirectional=False
        )

    def forward(self, x, cond=None):
        """
        when cond is None, modulation is only a function of x
        """
        # x = [batch, nchannels, length]
        # cond = [batch, nparams]
        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[2] % self.block_size) != 0:
            padding_size = self.block_size - (x_in_shape[2] % self.block_size)
            x = torch.nn.functional.pad(x, (0, padding_size))

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        # x_down = self.pool(torch.abs(x))
        x_down = self.pool(x)

        # upsample params [batch, nparams, nsteps]
        if self.cond_dim > 0 and cond is not None:
            cond_up = cond.unsqueeze(-1).repeat(1, 1, nsteps)

            # concat along channel dim [batch, nchannels+nparams, nsteps]
            x_down = torch.cat((x_down, cond_up), dim=1)

        # shape for LSTM [length, batch, nchannels]
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        if self.hidden_state is None:  # state was reset
            # init hidden and cell states with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.nfeatures * 2).requires_grad_().type_as(x)
            c0 = torch.zeros(self.num_layers, x.size(0), self.nfeatures * 2).requires_grad_().type_as(x)
            x_norm, self.hidden_state = self.lstm(x_down, (h0.detach(), c0.detach()))  # detach for truncated BPTT
        else:
            x_norm, self.hidden_state = self.lstm(x_down, self.hidden_state)

        # put shape back [batch, channels, length]
        x_norm = x_norm.permute(1, 2, 0)

        # reshape input and modulation sequence into blocks
        # [batch, channels, nsteps, block_size]
        x_in = torch.reshape(x, shape=(-1, self.nfeatures, nsteps, self.block_size))
        # [batch, channels*2, nsteps, 1]
        x_norm = torch.reshape(x_norm, shape=(-1, self.nfeatures * 2, nsteps, 1))

        # split modulation sequence along channels
        g, b = torch.chunk(x_norm, 2, dim=1)

        # modulate
        x_out = (x_in * g) + b

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        # crop to original (input) shape
        x_out = x_out[..., : x_in_shape[2]]

        return x_out

    def reset_state(self):
        self.hidden_state = None


# -----------------------------------------------------------------------------
# Tiny Temporal FiLM layer
# -----------------------------------------------------------------------------


class TinyTFiLM(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond,
    modulates x over time.

    nfeatures: number of features (i.e., convolution channels)
    cond_dim: number of control parameters, if cond_dim = 0, modulation is only a function of features
    block_size: size of blocks to modulate
    num_layers: number of LSTM layers
    """

    def __init__(self, nfeatures, bottleneck_dim, cond_dim, block_size, num_layers):
        super(TinyTFiLM, self).__init__()
        self.nfeatures = nfeatures
        self.bottleneck_dim = bottleneck_dim
        self.cond_dim = cond_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.hidden_state = None  # (hidden_state, cell_state)

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(kernel_size=block_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        # used for bottleneck
        self.conv = torch.nn.Conv1d(nfeatures, bottleneck_dim, kernel_size=1)

        self.lstm = torch.nn.LSTM(
            input_size=bottleneck_dim + cond_dim, hidden_size=bottleneck_dim, num_layers=num_layers, batch_first=False, bidirectional=False
        )

        # used to scale back up to nfeatures*2
        self.adaptor = torch.nn.Sequential(
            torch.nn.Conv1d(bottleneck_dim, 16, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, nfeatures * 2, kernel_size=1),
            torch.nn.ReLU(),
        )

    def forward(self, x, cond=None):
        """
        when cond is None, modulation is only a function of x
        """
        # x = [batch, nchannels, length]
        # cond = [batch, nparams]
        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[2] % self.block_size) != 0:
            padding_size = self.block_size - (x_in_shape[2] % self.block_size)
            x = torch.nn.functional.pad(x, (0, padding_size))

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        x_down = self.pool(x)

        # bottleneck
        x_down = self.conv(x_down)

        # upsample params [batch, nparams, nsteps]
        if self.cond_dim > 0 and cond is not None:
            cond_up = cond.unsqueeze(-1).repeat(1, 1, nsteps)

            # concat along channel dim [batch, nchannels+nparams, nsteps]
            x_down = torch.cat((x_down, cond_up), dim=1)

        # shape for LSTM [length, batch, nchannels]
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        if self.hidden_state is None:  # state was reset
            # init hidden and cell states with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.bottleneck_dim).requires_grad_().type_as(x)
            c0 = torch.zeros(self.num_layers, x.size(0), self.bottleneck_dim).requires_grad_().type_as(x)
            x_norm, self.hidden_state = self.lstm(x_down, (h0.detach(), c0.detach()))  # detach for truncated BPTT
        else:
            x_norm, self.hidden_state = self.lstm(x_down, self.hidden_state)

        # put shape back [batch, channels, length]
        x_norm = x_norm.permute(1, 2, 0)

        # mlp adaptor
        x_norm = self.adaptor(x_norm)

        # reshape input and modulation sequence into blocks
        # [batch, channels, nsteps, block_size]
        x_in = torch.reshape(x, shape=(-1, self.nfeatures, nsteps, self.block_size))
        # [batch, channels*2, nsteps, 1]
        x_norm = torch.reshape(x_norm, shape=(-1, self.nfeatures * 2, nsteps, 1))

        # split modulation sequence along channels
        g, b = torch.chunk(x_norm, 2, dim=1)

        # modulate
        x_out = (x_in * g) + b

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        # crop to original (input) shape
        x_out = x_out[..., : x_in_shape[2]]

        return x_out

    def reset_state(self):
        self.hidden_state = None


# -----------------------------------------------------------------------------
# Time-Varying FiLM - modulation layer
# -----------------------------------------------------------------------------


class TVFiLMMod(torch.nn.Module):
    """
    Given an input sequence x and a conditioning sequence cond_seq,
    returns a modulated sequence.

    nfeatures: number of input channels
    cond_dim: number of channels for conditioning sequence
    block_size: size of blocks to modulate
    """

    def __init__(self, nfeatures, cond_dim, block_size):
        super(TVFiLMMod, self).__init__()
        self.nfeatures = nfeatures
        self.cond_dim = cond_dim
        self.block_size = block_size

        self.adaptor = torch.nn.Conv1d(cond_dim, nfeatures * 2, kernel_size=1)

    def forward(self, x, cond_seq):
        # x = [batch, channels, length]
        # cond_seq = [batch, cond_dim, length]

        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[-1] % self.block_size) != 0:
            padding = self.block_size - (x_in_shape[-1] % self.block_size)
            x = torch.nn.functional.pad(x, [0, padding])

        x_pad_shape = x.shape
        nsteps = int(x_pad_shape[-1] / self.block_size)

        cond_seq = self.adaptor(cond_seq)
        cond_seq = cond_seq[..., -nsteps:]  # crop to match input length
        assert nsteps == cond_seq.shape[-1]

        # reshape input and conditioning sequence into blocks
        # [batch, channels, nsteps, block_size]
        x_in = torch.reshape(x, shape=(-1, self.nfeatures, nsteps, self.block_size))
        # [batch, channels*2, nsteps, 1]
        cond_seq = torch.reshape(cond_seq, shape=(-1, self.nfeatures * 2, nsteps, 1))

        # split modulation sequence along channels
        g, b = torch.chunk(cond_seq, 2, dim=1)

        # modulate
        x_out = (x_in * g) + b

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_pad_shape))

        # crop to original (input) shape
        x_out = x_out[..., : x_in_shape[2]]

        return x_out


# -----------------------------------------------------------------------------
# Time-Varying FiLM - conditioning layer
# -----------------------------------------------------------------------------


class TVFiLMCond(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond,
    returns a conditioning sequence that modulates FiLM layers
    over time.

    input_dim: number of input channels
    output_dim: number of channels for conditioning sequence
    cond_dim: number of control parameters, if cond_dim = 0, modulation is only a function of x
    block_size: size of blocks to modulate
    num_layers: number of LSTM layers
    """

    def __init__(self, input_dim, output_dim, cond_dim, block_size, num_layers):
        super(TVFiLMCond, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.block_size = block_size
        self.num_layers = num_layers

        # initialized as a tensor for torchscript tracing
        self.hidden_state = (
            torch.zeros(
                1,
            ),
            torch.zeros(
                1,
            ),
        )
        self.is_hidden_state_init = False

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(kernel_size=block_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.lstm = torch.nn.LSTM(
            input_size=input_dim + cond_dim, hidden_size=output_dim, num_layers=num_layers, batch_first=False, bidirectional=False
        )

    def forward(self, x, cond=None):
        """
        when cond is None, modulation is only a function of x
        """
        # x = [batch, nchannels, length]
        # cond = [batch, nparams]

        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[-1] % self.block_size) != 0:
            padding = self.block_size - (x_in_shape[-1] % self.block_size)
            x = torch.nn.functional.pad(x, [0, padding])

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        x_down = self.pool(torch.abs(x))

        # upsample params [batch, nparams, nsteps]
        if self.cond_dim > 0 and cond is not None:
            cond_up = cond.unsqueeze(-1).repeat(1, 1, nsteps)

            # concat along channel dim [batch, nchannels+nparams, nsteps]
            x_down = torch.cat((x_down, cond_up), dim=1)

        # shape for LSTM [length, batch, channels]
        x_down = x_down.permute(2, 0, 1)

        # conditioning sequence
        if self.is_hidden_state_init:
            cond_seq, new_hidden_state = self.lstm(x_down, self.hidden_state)
        else:  # state was reset
            # default to zeros if hidden state is not provided
            cond_seq, new_hidden_state = self.lstm(x_down)

        # put shape back [batch, channels, length]
        cond_seq = cond_seq.permute(1, 2, 0)

        self.update_state(new_hidden_state)
        return cond_seq  # [batch, output_dim, nsteps]

    def update_state(self, new_hidden):
        self.hidden_state = new_hidden
        self.is_hidden_state_init = True

    def detach_state(self) -> None:
        if self.is_hidden_state_init:
            # TODO(cm): check whether clone is required or not
            self.hidden_state = tuple((h.detach().clone() for h in self.hidden_state))

    def reset_state(self):
        self.is_hidden_state_init = False


# -----------------------------------------------------------------------------
# TCN Conditional Block
# -----------------------------------------------------------------------------


class TCNCondBlock(torch.nn.Module):
    """
    cond. types: None, FiLM, TFiLM, TVFiLM

    cond_dim:   if None -> cond_dim = 0
                if film -> cond_dim = conditioning MLP output size
                if tfilm -> cond_dim = number of control parameters
                if ttfilm -> cond_dim = number of control parameters
                if tvfilm -> cond_dim = TVFiLMCond output_dim
    batchnorm:  available only for non-conditional models

    activation types: Tanh, PReLU, Rational
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        causal,
        batchnorm,
        residual,
        kernel_size,
        padding,
        dilation,
        groups,
        bias,
        cond_type,
        cond_dim,
        cond_block_size,
        cond_num_layers,
        act_type,
    ):
        super(TCNCondBlock, self).__init__()
        assert cond_dim >= 0
        assert cond_type in [None, "film", "tfilm", "ttfilm", "tvfilm"]
        assert cond_block_size > 0
        assert cond_num_layers > 0
        assert act_type in ["tanh", "prelu", "rational"]

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.causal = causal
        self.batchnorm = batchnorm
        self.residual = residual
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.cond_type = cond_type
        self.cond_dim = cond_dim
        self.cond_block_size = cond_block_size
        self.cond_num_layers = cond_num_layers
        self.act_type = act_type

        # CONV
        self.conv = torch.nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups, bias=bias)

        # CONDITIONING/MODULATION
        if cond_type == "film":
            self.film = FiLM(nfeatures=out_ch, cond_dim=cond_dim)
        elif cond_type == "tfilm":
            self.film = TFiLM(nfeatures=out_ch, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers)
        elif cond_type == "ttfilm":
            self.film = TinyTFiLM(
                nfeatures=out_ch, bottleneck_dim=4, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers
            )
        elif cond_type == "tvfilm":
            self.film = TVFiLMMod(nfeatures=out_ch, cond_dim=cond_dim, block_size=cond_block_size)
        elif cond_type is None and batchnorm:
            self.bn = torch.nn.BatchNorm1d(out_ch)

        # ACTIVATIONS
        if act_type == "tanh":
            self.act = torch.nn.Tanh()
        elif act_type == "prelu":
            self.act = torch.nn.PReLU(num_parameters=out_ch)
        elif act_type == "rational":
            self.act = Rational(approx_func="tanh", degrees=[4, 3], version="A")

        # RESIDUAL
        if residual:
            self.res = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False)

    def forward(self, x, cond=None):
        """
        cond:   if film -> cond = conditioning MLP output
                if tfilm -> cond = control parameters
                if tvfilm -> cond = TVFiLMCond conditioning sequence
        """

        x_in = x

        # CONV
        x = self.conv(x)

        # CONDITIONING/MODULATION
        if self.cond_type is not None:
            x = self.film(x, cond)
        elif self.cond_type is None and self.batchnorm:
            x = self.bn(x)

        # ACTIVATIONS
        x = self.act(x)

        # OUTPUT
        if self.residual:
            x_res = self.res(x_in)

            if self.causal:
                x = x + causal_crop(x_res, x.shape[-1])
            else:
                x = x + center_crop(x_res, x.shape[-1])

        return x


# -----------------------------------------------------------------------------
# GCN Conditional Block
# -----------------------------------------------------------------------------


class GCNCondBlock(torch.nn.Module):
    """
    cond. types: None, FiLM, TFiLM, TVFiLM

    cond_dim:   if None -> cond_dim = 0
                if film -> cond_dim = conditioning MLP output size
                if tfilm -> cond_dim = number of control parameters
                if ttfilm -> cond_dim = number of control parameters
                if tvfilm -> cond_dim = TVFiLMCond output_dim
    batchnorm:  available only for non-conditional models
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        causal,
        batchnorm,
        residual,
        kernel_size,
        padding,
        dilation,
        groups,
        bias,
        cond_type,
        cond_dim,
        cond_block_size,
        cond_num_layers,
    ):
        super(GCNCondBlock, self).__init__()
        assert cond_dim >= 0
        assert cond_type in [None, "film", "tfilm", "ttfilm", "tvfilm"]
        assert cond_block_size > 0

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.causal = causal
        self.batchnorm = batchnorm
        self.residual = residual
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.cond_type = cond_type
        self.cond_dim = cond_dim
        self.cond_block_size = cond_block_size
        self.cond_num_layers = cond_num_layers

        # CONV
        self.conv = torch.nn.Conv1d(
            in_ch,
            out_ch * 2,  # for the gated activation
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # CONDITIONING/MODULATION
        if cond_type == "film":
            self.film = FiLM(nfeatures=out_ch * 2, cond_dim=cond_dim)
        elif cond_type == "tfilm":
            self.film = TFiLM(nfeatures=out_ch * 2, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers)
        elif cond_type == "ttfilm":
            self.film = TinyTFiLM(
                nfeatures=out_ch * 2, bottleneck_dim=4, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers
            )
        elif cond_type == "tvfilm":
            self.film = TVFiLMMod(nfeatures=out_ch * 2, cond_dim=cond_dim, block_size=cond_block_size)
        elif cond_type is None and batchnorm:
            self.bn = torch.nn.BatchNorm1d(out_ch * 2)

        # GATED ACTIVATION
        self.tanh = torch.nn.Tanh()
        self.sigm = torch.nn.Sigmoid()

        # MIX
        self.mix = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x, cond=None):
        """
        cond:   if film -> cond = conditioning MLP output
                if tfilm -> cond = control parameters
                if tvfilm -> cond = TVFiLMCond conditioning sequence
        """

        x_res = x

        # CONV
        y = self.conv(x)

        # CONDITIONING/MODULATION
        if self.cond_type is not None:
            y = self.film(y, cond)
        elif self.cond_type is None and self.batchnorm:
            y = self.bn(y)

        # GATED ACTIVATION
        z = self.tanh(y[:, : self.out_ch, :]) * self.sigm(y[:, self.out_ch :, :])

        # OUTPUT
        if self.residual:
            if self.causal:
                x = self.mix(z) + causal_crop(x_res, z.shape[-1])
            else:
                x = self.mix(z) + center_crop(x_res, z.shape[-1])
        else:
            x = self.mix(z)

        return x, z


# -----------------------------------------------------------------------------
# Diagonal State-Space Model Block
# -----------------------------------------------------------------------------

c2r = torch.view_as_real
r2c = torch.view_as_complex


class DSSM(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float = 1e-3,
    ):
        super().__init__()

        H = input_dim
        self.H = H
        N = state_dim
        self.N = N

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.register("log_dt", log_dt, lr)

        C = torch.randn(H, N, dtype=torch.cfloat)
        self.C = torch.nn.Parameter(c2r(C))

        log_A_real = torch.log(0.5 * torch.ones(H, N))
        A_imag = math.pi * repeat(torch.arange(N), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        self.D = torch.nn.Parameter(torch.randn(H))

    def get_kernel(self, length: int):  # `length` is `L`
        dt = torch.exp(self.log_dt)  # (H)
        C = r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        P = dtA.unsqueeze(-1) * torch.arange(length, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(P)).real
        return K

    def register(self, name: str, tensor: Tensor, lr: float = None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, torch.nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}  # Never use weight decay
            if lr is not None:  # Use custom learning rate when a learning rate is given
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, u: Tensor, length: Tensor = None):
        # Input and output shape (B H L)
        assert u.dim() == 3
        B, H, L = u.size()
        assert H == self.H

        # length shape (L)
        if length is None:
            length = torch.empty(B).fill_(L)
        assert length.dim() == 1
        assert length.size(0) == B
        assert torch.all(length <= L)
        length = length.to(torch.long).cpu()

        l_s, i_s = length.sort(stable=True, descending=True)
        l_s, i_s = l_s.tolist(), i_s.tolist()
        prev_i = 0
        pair: list[tuple[int, list[int]]] = []
        for i in range(1, B):
            if l_s[i] == l_s[i - 1]:
                continue
            pair.append((l_s[prev_i], i_s[prev_i:i]))
            prev_i = i
        pair.append((l_s[prev_i], i_s[prev_i:]))

        kernel = self.get_kernel(L)  # (H L)
        out = torch.zeros_like(u)  # (B H L)

        for l, idxs in pair:
            _k = kernel[:, :l]
            _u = u[idxs, :, :l]

            k_f = torch.fft.rfft(_k, n=2 * l)  # (H l)
            u_f = torch.fft.rfft(_u, n=2 * l)  # (B H l)
            y = torch.fft.irfft(u_f * k_f, n=2 * l)[..., :l]  # (B H l)
            y += _u * self.D.unsqueeze(-1)  # (B H l)

            out[idxs, :, :l] = y[...]

        return out


# -----------------------------------------------------------------------------
# S4 Conditional Block
# -----------------------------------------------------------------------------


class S4CondBlock(torch.nn.Module):
    def __init__(
        self,
        channel_width: int,
        batchnorm: bool,
        residual: bool,
        s4_state_dim: int,
        s4_learning_rate: float,
        cond_type: str,
        cond_dim: int,
        cond_block_size: int,
        cond_num_layers: int,
        act_type: str,
    ):
        super(S4CondBlock, self).__init__()
        assert cond_dim >= 0
        assert cond_type in [None, "film", "tfilm", "ttfilm", "tvfilm"]
        assert cond_block_size > 0
        assert cond_num_layers > 0
        assert act_type in ["tanh", "prelu", "rational"]

        self.channel_width = channel_width
        self.batchnorm = batchnorm
        self.residual = residual
        self.s4_state_dim = s4_state_dim
        self.s4_learning_rate = s4_learning_rate
        self.cond_type = cond_type
        self.cond_dim = cond_dim
        self.cond_block_size = cond_block_size
        self.cond_num_layers = cond_num_layers
        self.act_type = act_type

        # LINEAR
        self.linear = torch.nn.Linear(channel_width, channel_width)

        # S4
        self.s4 = DSSM(input_dim=channel_width, state_dim=s4_state_dim, lr=s4_learning_rate)

        # CONDITIONING/MODULATION
        if cond_type == "film":
            self.film = FiLM(nfeatures=channel_width, cond_dim=cond_dim)
        elif cond_type == "tfilm":
            self.film = TFiLM(nfeatures=channel_width, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers)
        elif cond_type == "ttfilm":
            self.film = TinyTFiLM(
                nfeatures=channel_width, bottleneck_dim=4, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers
            )
        elif cond_type == "tvfilm":
            self.film = TVFiLMMod(nfeatures=channel_width, cond_dim=cond_dim, block_size=cond_block_size)
        elif cond_type is None and batchnorm:
            self.bn = torch.nn.BatchNorm1d(channel_width)

        # ACTIVATIONS
        if act_type == "tanh":
            self.act1 = torch.nn.Tanh()
            self.act2 = torch.nn.Tanh()
        elif act_type == "prelu":
            self.act1 = torch.nn.PReLU(num_parameters=channel_width)
            self.act2 = torch.nn.PReLU(num_parameters=channel_width)
        elif act_type == "rational":
            self.act1 = Rational(approx_func="tanh", degrees=[4, 3], version="A")
            self.act2 = Rational(approx_func="tanh", degrees=[4, 3], version="A")

        # RESIDUAL
        if residual:
            self.res = torch.nn.Conv1d(
                channel_width,
                channel_width,
                kernel_size=1,
                groups=channel_width,
                bias=False,
            )

    def forward(self, x: Tensor, cond: Tensor = None) -> Tensor:

        x_in = x

        # LINEAR
        x = rearrange(x, "B H L -> B L H")
        x = self.linear(x)
        x = rearrange(x, "B L H -> B H L")

        # ACTIVATION
        x = self.act1(x)

        # S4
        x = self.s4(x)

        # CONDITIONING/MODULATION
        if self.cond_type is not None:
            x = self.film(x, cond)
        elif self.cond_type is None and self.batchnorm:
            x = self.bn(x)

        # ACTIVATION
        x = self.act2(x)

        # OUTPUT
        if self.residual:
            x = x + self.res(x_in)

        return x
