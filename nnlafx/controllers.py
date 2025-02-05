import torch
from typing import Union
from nnlafx.modules import MLP


# -----------------------------------------------------------------------------
# Dummy Controller
# -----------------------------------------------------------------------------


class DummyController(torch.nn.Module):
    """Dummy controller that returns None"""

    def __init__(self):
        super().__init__()
        self.num_controls = 0
        self.num_control_params = 0

    def forward(self, x: torch.Tensor):
        return None


# -----------------------------------------------------------------------------
# Static Controller
# -----------------------------------------------------------------------------


class StaticController(torch.nn.Module):
    """Non-conditional controller that maps an internal tensor
    to a set of control parameters.
    """

    def __init__(
        self,
        num_control_params: int,
        control_params_initial: Union[str, float],
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_controls = 0
        self.num_control_params = num_control_params
        if control_params_initial == "rnd":
            self.control_params = torch.nn.Parameter(torch.randn(num_control_params))
        else:
            self.control_params = torch.nn.Parameter(torch.ones(num_control_params) * control_params_initial)
        self.lr_multiplier = lr_multiplier

        self.act = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        bs, chs, seq_len = x.size()
        return self.act(self.control_params).unsqueeze(0).unsqueeze(-1).repeat(bs, 1, 1)


# -----------------------------------------------------------------------------
# Static Conditional Controller
# -----------------------------------------------------------------------------


class StaticCondController(torch.nn.Module):
    """Conditional controller that maps input controls
    to a set of control parameters.
    """

    def __init__(
        self,
        num_controls: int,
        num_control_params: int,
        num_layers: int = 3,
        hidden_dim: int = 16,
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_controls = num_controls
        self.num_control_params = num_control_params
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lr_multiplier = lr_multiplier

        self.projection = MLP(
            input_dim=num_controls,
            output_dim=num_control_params,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=torch.nn.Sigmoid(),
        )

    def forward(self, controls: torch.Tensor):
        bs, chs = controls.shape
        assert chs == self.num_controls

        return self.projection(controls).unsqueeze(-1)


# -----------------------------------------------------------------------------
#  Dynamic Controller
# -----------------------------------------------------------------------------


class DynamicController(torch.nn.Module):
    """Non-conditional controller that maps an input signal
    to a set of time-varying control parameters.
    """

    def __init__(
        self,
        num_control_params: int,
        block_size: int = 128,
        num_layers: int = 1,
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_controls = 0
        self.num_control_params = num_control_params
        self.block_size = block_size
        self.num_layers = num_layers
        self.lr_multiplier = lr_multiplier

        self.hidden_state = (
            torch.zeros(
                1,
            ),
            torch.zeros(
                1,
            ),
        )  # initialized as a tensor for torchscript tracing
        self.is_hidden_state_init = False

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(
            kernel_size=block_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=num_control_params,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )

        self.act = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        bs, chs, seq_len = x.shape
        assert chs == 1

        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (seq_len % self.block_size) != 0:
            padding_size = self.block_size - (seq_len % self.block_size)
            x = torch.nn.functional.pad(x, (0, padding_size))

        nsteps = int(x.shape[-1] / self.block_size)

        # downsample signal
        x_down = self.pool(torch.abs(x))

        # shape for LSTM [length, batch, nchannels]
        x_down = x_down.permute(2, 0, 1)

        # control sequence
        if self.is_hidden_state_init:
            control_params, new_hidden_state = self.lstm(x_down, self.hidden_state)
        else:  # state was reset
            control_params, new_hidden_state = self.lstm(x_down)

        # put shape back [batch, channels, length]
        control_params = control_params.permute(1, 2, 0)

        # limit
        control_params = self.act(control_params)

        # upsample to original size
        control_params = control_params.repeat_interleave(self.block_size, dim=-1)
        self.update_state(new_hidden_state)
        return control_params[..., :seq_len]

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        self.is_hidden_state_init = False

    def detach_states(self) -> None:
        self.detach_state()

    def detach_state(self) -> None:
        if self.is_hidden_state_init:
            self.hidden_state = tuple((h.detach() for h in self.hidden_state))

    def update_state(self, new_hidden):
        self.hidden_state = new_hidden
        self.is_hidden_state_init = True


# -----------------------------------------------------------------------------
# Dynamic Conditional Controller
# -----------------------------------------------------------------------------


class DynamicCondController(torch.nn.Module):
    """Conditional controller that maps input signal and input controls
    to a set of time-varying control parameters.
    """

    def __init__(
        self,
        num_controls: int,
        num_control_params: int,
        block_size: int = 128,
        num_layers: int = 1,
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_controls = num_controls
        self.num_control_params = num_control_params
        self.block_size = block_size
        self.num_layers = num_layers
        self.lr_multiplier = lr_multiplier

        self.hidden_state = (
            torch.zeros(
                1,
            ),
            torch.zeros(
                1,
            ),
        )  # initialized as a tensor for torchscript tracing
        self.is_hidden_state_init = False

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(
            kernel_size=block_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        self.lstm = torch.nn.LSTM(
            input_size=1 + num_controls,
            hidden_size=num_control_params,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )

        self.act = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, controls: torch.Tensor):
        bs_x, chs_x, seq_len_x = x.shape
        bs_c, chs_c = controls.shape
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_controls

        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (seq_len_x % self.block_size) != 0:
            padding_size = self.block_size - (seq_len_x % self.block_size)
            x = torch.nn.functional.pad(x, (0, padding_size))

        nsteps = int(x.shape[-1] / self.block_size)

        # downsample signal
        x_down = self.pool(torch.abs(x))

        # upsample params [batch, nparams, nsteps]
        controls_up = controls.unsqueeze(-1).repeat(1, 1, nsteps)

        # concat along channel dim [batch, nchannels+nparams, nsteps]
        x_down = torch.cat((x_down, controls_up), dim=1)

        # shape for LSTM [length, batch, nchannels]
        x_down = x_down.permute(2, 0, 1)

        # control sequence
        if self.is_hidden_state_init:
            control_params, new_hidden_state = self.lstm(x_down, self.hidden_state)
        else:  # state was reset
            control_params, new_hidden_state = self.lstm(x_down)

        # put shape back [batch, channels, length]
        control_params = control_params.permute(1, 2, 0)

        # limit
        control_params = self.act(control_params)

        # upsample to original size
        control_params = control_params.repeat_interleave(self.block_size, dim=-1)
        self.update_state(new_hidden_state)
        return control_params[..., :seq_len_x]

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        self.is_hidden_state_init = False

    def detach_states(self) -> None:
        self.detach_state()

    def detach_state(self) -> None:
        if self.is_hidden_state_init:
            self.hidden_state = tuple((h.detach() for h in self.hidden_state))

    def update_state(self, new_hidden):
        self.hidden_state = new_hidden
        self.is_hidden_state_init = True
