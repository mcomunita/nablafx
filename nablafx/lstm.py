import torch

from nablafx.modules import TVFiLMCond


# -----------------------------------------------------------------------------
# LSTM (Conditional)
# -----------------------------------------------------------------------------


class LSTM(torch.nn.Module):
    """
    cond_type:  if None -> basic LSTM, no modulation, no parametric control
                if fixed -> parametric control. cond_dim = number of control parameters
                if tvcond -> time-varying conditioning. cond_dim = any size
    """

    def __init__(
        self,
        num_inputs=1,
        num_outputs=1,
        num_controls=0,
        hidden_size=32,
        num_layers=1,
        residual=False,
        direct_path=False,
        cond_type=None,
        cond_block_size=128,
        cond_num_layers=1,
    ):
        super().__init__()

        assert cond_type in [None, "fixed", "tvcond"]
        if cond_type == "fixed":
            assert num_controls > 0
        if cond_type == "tvcond":
            assert num_controls >= 0  # tvcond can be used for parametric and non-parametric models

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_controls = num_controls
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.residual = residual
        self.direct_path = direct_path
        self.cond_type = cond_type
        if cond_type in [None, "fixed"]:  # no conditioning or fixed parametric conditioning
            self.cond_dim = 0
        elif cond_type == "tvcond":  # time-varying conditioning
            self.cond_dim = 16
        self.cond_block_size = cond_block_size
        self.cond_num_layers = cond_num_layers

        self.hidden_state = (
            torch.zeros(
                1,
            ),
            torch.zeros(
                1,
            ),
        )  # initialized as a tensor for torchscript tracing
        self.is_hidden_state_init = False

        # CONDITIONING
        if cond_type == "tvcond":
            self.cond_nn = TVFiLMCond(
                input_dim=num_inputs,
                output_dim=self.cond_dim,
                cond_dim=num_controls,
                block_size=cond_block_size,
                num_layers=cond_num_layers,
            )

        # DIRECT PATH
        if direct_path:
            self.direct_gain = torch.nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)  # direct path gain

        # BLOCKS
        if cond_type is None:
            input_size = num_inputs
        elif cond_type == "fixed":
            input_size = num_inputs + num_controls
        elif cond_type == "tvcond":
            input_size = num_inputs + self.cond_dim

        self.lstm = torch.nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=False, bidirectional=False)

        self.lin = torch.nn.Linear(self.hidden_size, self.num_outputs)

    def forward(self, x, p=None):
        # x = input : [batch, num_inputs, length]
        # p = params : [batch, num_controls]

        # DIRECT PATH
        if self.direct_path:
            x_direct = torch.tanh(self.direct_gain(x))
            x_direct = x_direct.permute(2, 0, 1)  # shape for LSTM (seq, batch, channel)

        # CONDITIONING
        s = x.size(-1)  # length

        if self.cond_type is None:
            pass
        elif self.cond_type == "fixed" and self.num_controls > 0 and p is not None:
            cond = p.unsqueeze(-1)  # [batch, num_controls, 1]
            cond = cond.repeat(1, 1, s)  # expand to every time step
            x = torch.cat((x, cond), dim=1)  # append to input along feature dim
        elif self.cond_type == "tvcond":
            cond = self.cond_nn(x, p)  # get conditioning sequence
            cond = cond.repeat_interleave(self.cond_block_size, dim=-1)  # upsample cond sequence
            cond = cond[:, :, :s]  # crop to length of input
            x = torch.cat((x, cond), dim=1)  # append to input along feature dim

        # PROCESSING PATH
        x = x.permute(2, 0, 1)  # shape for LSTM (seq, batch, channel)

        if self.is_hidden_state_init:
            x_proc, new_hidden_state = self.lstm(x, self.hidden_state)
        else:  # state was reset
            x_proc, new_hidden_state = self.lstm(x)

        if self.residual:
            res = x[:, :, :1]  # [length, batch, num_inputs]
            x_proc = self.lin(x_proc) + res
        else:
            x_proc = self.lin(x_proc)

        # SUM SIGNALS
        if self.direct_path:
            out = torch.tanh(x_direct + x_proc)
        else:
            out = torch.tanh(x_proc)

        out = out.permute(1, 2, 0)  # put shape back (batch, channel, seq)
        self.update_state(new_hidden_state)
        return out

    def reset_states(self):
        """
        reset state for LSTM and conditioning layers
        """
        self.reset_state()

        for layer in self.modules():
            if isinstance(layer, TVFiLMCond):
                layer.reset_state()

    def reset_state(self):
        self.is_hidden_state_init = False

    def detach_states(self) -> None:
        """
        detach state for LSTM and conditioning layers
        for truncated back-propagation through time
        """
        self.detach_state()

        for layer in self.modules():
            if isinstance(layer, TVFiLMCond):
                layer.detach_state()

    def detach_state(self) -> None:
        if self.is_hidden_state_init:
            self.hidden_state = tuple((h.detach() for h in self.hidden_state))

    def update_state(self, new_hidden):
        self.hidden_state = new_hidden
        self.is_hidden_state_init = True
