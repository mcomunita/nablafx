import torch
from rational.torch import Rational
from typing import Tuple

from nablafx.dsp import denormalize_parameters, biquad, sosfilt, sosfilt_via_fsm, lfilter_via_fsm
from nablafx.siren import Modulator, SirenNet


# -----------------------------------------------------------------------------
# BASIC
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Phase Shift (WIP)
# -----------------------------------------------------------------------------


class PhaseShift(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_shift: float = -180.0,
        max_shift: float = +180.0,
        control_type: str = "static",
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {"shift": (min_shift, max_shift)}
        self.control_type = control_type
        self.num_control_params = 1

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {"shift": params[:, 0, :]}
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x
        param_dict = self.get_param_dict(control_params)
        output = self.process(x, **param_dict, train=train)
        return output, param_dict

    def process(self, x: torch.Tensor, shift: torch.Tensor, train: bool = False):
        bs, chs, seq_len = x.size()
        shift_rad = shift.view(bs, chs, -1) * (torch.pi / 180.0)
        shift_rad = shift_rad.repeat(1, 1, seq_len)
        return x * torch.cos(shift_rad) + torch.roll(x, 1, dims=-1) * torch.sin(shift_rad)


# -----------------------------------------------------------------------------
# Phase Invertion
# -----------------------------------------------------------------------------


class PhaseInversion(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_control_params = 0
        self.control_type = None

    def get_param_dict(self, params: torch.Tensor):
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train), self.get_param_dict(control_params)

    def process(self, x: torch.Tensor, train: bool = False):
        return -x


# -----------------------------------------------------------------------------
# Gain
# -----------------------------------------------------------------------------


class Gain(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -32.0,
        max_gain_db: float = 32.0,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {"gain_db": (min_gain_db, max_gain_db)}
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 1

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {"gain_db": params[:, 0, :]}
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x
        param_dict = self.get_param_dict(control_params)
        output = self.process(x, **param_dict, train=train)
        return output, param_dict

    def process(self, x: torch.Tensor, gain_db: torch.Tensor, train: bool = False):
        bs, chs, seq_len = x.size()
        gain_ln = 10 ** (gain_db.view(bs, chs, -1) / 20.0)
        return x * gain_ln


# -----------------------------------------------------------------------------
# DC Offset
# -----------------------------------------------------------------------------


class DCOffset(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_offset: float = -2.0,
        max_offset: float = +2.0,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {"offset": (min_offset, max_offset)}
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 1

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {"offset": params[:, 0, :]}
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x
        param_dict = self.get_param_dict(control_params)
        output = self.process(x, **param_dict, train=train)
        return output, param_dict

    def process(self, x: torch.Tensor, offset: torch.Tensor, train: bool = False):
        bs, chs, seq_len = x.size()
        return x + offset.view(bs, chs, -1)


# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Parametric EQ
# -----------------------------------------------------------------------------


class ParametricEQ(torch.nn.Module):
    """Parametric EQ with low-shelving, 3 peakings, and high-shelving filters."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.param_ranges = {
            "low_shelf_gain_db": (min_gain_db, max_gain_db),
            "low_shelf_cutoff_freq": (20.0, 2000.0),
            "low_shelf_q_factor": (min_q_factor, max_q_factor),
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (20.0, 200.0),
            "band0_q_factor": (min_q_factor, max_q_factor),
            "band1_gain_db": (min_gain_db, max_gain_db),
            "band1_cutoff_freq": (200.0, 2000.0),
            "band1_q_factor": (min_q_factor, max_q_factor),
            "band2_gain_db": (min_gain_db, max_gain_db),
            "band2_cutoff_freq": (2000.0, 12000.0),
            "band2_q_factor": (min_q_factor, max_q_factor),
            "high_shelf_gain_db": (min_gain_db, max_gain_db),
            "high_shelf_cutoff_freq": (4000.0, 16000.0),
            "high_shelf_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 15

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {
            "low_shelf_gain_db": params[:, 0, :],
            "low_shelf_cutoff_freq": params[:, 1, :],
            "low_shelf_q_factor": params[:, 2, :],
            "band0_gain_db": params[:, 3, :],
            "band0_cutoff_freq": params[:, 4, :],
            "band0_q_factor": params[:, 5, :],
            "band1_gain_db": params[:, 6, :],
            "band1_cutoff_freq": params[:, 7, :],
            "band1_q_factor": params[:, 8, :],
            "band2_gain_db": params[:, 9, :],
            "band2_cutoff_freq": params[:, 10, :],
            "band2_q_factor": params[:, 11, :],
            "high_shelf_gain_db": params[:, 12, :],
            "high_shelf_cutoff_freq": params[:, 13, :],
            "high_shelf_q_factor": params[:, 14, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)
        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        low_shelf_gain_db: torch.Tensor,
        low_shelf_cutoff_freq: torch.Tensor,
        low_shelf_q_factor: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        band1_gain_db: torch.Tensor,
        band1_cutoff_freq: torch.Tensor,
        band1_q_factor: torch.Tensor,
        band2_gain_db: torch.Tensor,
        band2_cutoff_freq: torch.Tensor,
        band2_q_factor: torch.Tensor,
        high_shelf_gain_db: torch.Tensor,
        high_shelf_cutoff_freq: torch.Tensor,
        high_shelf_q_factor: torch.Tensor,
        train: bool = False,
    ):
        # since we are learning parameter we compute coefficients on the fly
        sos = self.compute_coefficients(
            self.sample_rate,
            low_shelf_gain_db,
            low_shelf_cutoff_freq,
            low_shelf_q_factor,
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
            band1_gain_db,
            band1_cutoff_freq,
            band1_q_factor,
            band2_gain_db,
            band2_cutoff_freq,
            band2_q_factor,
            high_shelf_gain_db,
            high_shelf_cutoff_freq,
            high_shelf_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        low_shelf_gain_db: torch.Tensor,
        low_shelf_cutoff_freq: torch.Tensor,
        low_shelf_q_factor: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        band1_gain_db: torch.Tensor,
        band1_cutoff_freq: torch.Tensor,
        band1_q_factor: torch.Tensor,
        band2_gain_db: torch.Tensor,
        band2_cutoff_freq: torch.Tensor,
        band2_q_factor: torch.Tensor,
        high_shelf_gain_db: torch.Tensor,
        high_shelf_cutoff_freq: torch.Tensor,
        high_shelf_q_factor: torch.Tensor,
    ):
        bs = low_shelf_gain_db.size(0)

        # five second order sections
        sos = torch.zeros(bs, 5, 6).type_as(low_shelf_gain_db)
        # ------------ low shelf ------------
        b, a = biquad(
            low_shelf_gain_db,
            low_shelf_cutoff_freq,
            low_shelf_q_factor,
            sample_rate,
            "low_shelf",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)
        # ------------ band0 ------------
        b, a = biquad(
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "peaking",
        )
        sos[:, 1, :] = torch.cat((b, a), dim=-1)
        # ------------ band1 ------------
        b, a = biquad(
            band1_gain_db,
            band1_cutoff_freq,
            band1_q_factor,
            sample_rate,
            "peaking",
        )
        sos[:, 2, :] = torch.cat((b, a), dim=-1)
        # ------------ band2 ------------
        b, a = biquad(
            band2_gain_db,
            band2_cutoff_freq,
            band2_q_factor,
            sample_rate,
            "peaking",
        )
        sos[:, 3, :] = torch.cat((b, a), dim=-1)
        # ------------ high shelf ------------
        b, a = biquad(
            high_shelf_gain_db,
            high_shelf_cutoff_freq,
            high_shelf_q_factor,
            sample_rate,
            "high_shelf",
        )
        sos[:, 4, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Shelving EQ
# -----------------------------------------------------------------------------


class ShelvingEQ(torch.nn.Module):
    """EQ with high-pass, low-shelving, high-shelving and low-pass filters."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.param_ranges = {
            "highpass_cutoff_freq": (20.0, 2000.0),
            "highpass_q_factor": (min_q_factor, max_q_factor),
            "low_shelf_gain_db": (min_gain_db, max_gain_db),
            "low_shelf_cutoff_freq": (20.0, 4000.0),
            "low_shelf_q_factor": (min_q_factor, max_q_factor),
            "high_shelf_gain_db": (min_gain_db, max_gain_db),
            "high_shelf_cutoff_freq": (4000.0, 24000.0),
            "high_shelf_q_factor": (min_q_factor, max_q_factor),
            "lowpass_cutoff_freq": (4000.0, 24000.0),
            "lowpass_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 10

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {
            "highpass_cutoff_freq": params[:, 0, :],
            "highpass_q_factor": params[:, 1, :],
            "low_shelf_gain_db": params[:, 2, :],
            "low_shelf_cutoff_freq": params[:, 3, :],
            "low_shelf_q_factor": params[:, 4, :],
            "high_shelf_gain_db": params[:, 5, :],
            "high_shelf_cutoff_freq": params[:, 6, :],
            "high_shelf_q_factor": params[:, 7, :],
            "lowpass_cutoff_freq": params[:, 8, :],
            "lowpass_q_factor": params[:, 9, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        highpass_cutoff_freq: torch.Tensor,
        highpass_q_factor: torch.Tensor,
        low_shelf_gain_db: torch.Tensor,
        low_shelf_cutoff_freq: torch.Tensor,
        low_shelf_q_factor: torch.Tensor,
        high_shelf_gain_db: torch.Tensor,
        high_shelf_cutoff_freq: torch.Tensor,
        high_shelf_q_factor: torch.Tensor,
        lowpass_cutoff_freq: torch.Tensor,
        lowpass_q_factor: torch.Tensor,
        train: bool = False,
    ):
        # since we are learning parameter we compute coefficients on the fly
        sos = self.compute_coefficients(
            self.sample_rate,
            highpass_cutoff_freq,
            highpass_q_factor,
            low_shelf_gain_db,
            low_shelf_cutoff_freq,
            low_shelf_q_factor,
            high_shelf_gain_db,
            high_shelf_cutoff_freq,
            high_shelf_q_factor,
            lowpass_cutoff_freq,
            lowpass_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        highpass_cutoff_freq: torch.Tensor,
        highpass_q_factor: torch.Tensor,
        low_shelf_gain_db: torch.Tensor,
        low_shelf_cutoff_freq: torch.Tensor,
        low_shelf_q_factor: torch.Tensor,
        high_shelf_gain_db: torch.Tensor,
        high_shelf_cutoff_freq: torch.Tensor,
        high_shelf_q_factor: torch.Tensor,
        lowpass_cutoff_freq: torch.Tensor,
        lowpass_q_factor: torch.Tensor,
    ):
        bs = low_shelf_gain_db.size(0)

        # four second order sections
        sos = torch.zeros(bs, 4, 6).type_as(highpass_cutoff_freq)
        # ------------ highpass ------------
        b, a = biquad(
            torch.zeros_like(highpass_cutoff_freq),  # gain_db
            highpass_cutoff_freq,
            highpass_q_factor,
            sample_rate,
            "highpass",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)
        # ------------ low shelf ------------
        b, a = biquad(
            low_shelf_gain_db,
            low_shelf_cutoff_freq,
            low_shelf_q_factor,
            sample_rate,
            "low_shelf",
        )
        sos[:, 1, :] = torch.cat((b, a), dim=-1)
        # ------------ high shelf ------------
        b, a = biquad(
            high_shelf_gain_db,
            high_shelf_cutoff_freq,
            high_shelf_q_factor,
            sample_rate,
            "high_shelf",
        )
        sos[:, 2, :] = torch.cat((b, a), dim=-1)
        # ------------ lowpass ------------
        b, a = biquad(
            torch.zeros_like(lowpass_cutoff_freq),  # gain_db
            lowpass_cutoff_freq,
            lowpass_q_factor,
            sample_rate,
            "lowpass",
        )
        sos[:, 3, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Peak/Notch
# -----------------------------------------------------------------------------


class Peaking(torch.nn.Module):
    """Peaking filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 20000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band0_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 3

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {
            "band0_gain_db": params[:, 0, :],
            "band0_cutoff_freq": params[:, 1, :],
            "band0_q_factor": params[:, 2, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
    ):
        bs = band0_gain_db.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(band0_gain_db)
        # ------------ band0 ------------
        b, a = biquad(
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "peaking",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Lowpass
# -----------------------------------------------------------------------------


class Lowpass(torch.nn.Module):
    """Lowpass filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_cutoff_freq: float = 2000.0,
        max_cutoff_freq: float = 20000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 2

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {
            "cutoff_freq": params[:, 0, :],
            "q_factor": params[:, 1, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        cutoff_freq: torch.Tensor,
        q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            cutoff_freq,
            q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        cutoff_freq: torch.Tensor,
        q_factor: torch.Tensor,
    ):
        bs = cutoff_freq.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(cutoff_freq)
        # ------------ band0 ------------
        b, a = biquad(
            torch.zeros_like(cutoff_freq),  # gain_db
            cutoff_freq,
            q_factor,
            sample_rate,
            "low_pass",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Highpass
# -----------------------------------------------------------------------------


class Highpass(torch.nn.Module):
    """Highpass filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "band0_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band0_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 2

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {
            "band0_cutoff_freq": params[:, 0, :],
            "band0_q_factor": params[:, 1, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            band0_cutoff_freq,
            band0_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
    ):
        bs = band0_cutoff_freq.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(band0_cutoff_freq)
        # ------------ band0 ------------
        b, a = biquad(
            torch.zeros_like(band0_cutoff_freq),  # gain_db
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "high_pass",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Lowshelf
# -----------------------------------------------------------------------------


class Lowshelf(torch.nn.Module):
    """Lowshelf filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band0_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 3

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {
            "band0_gain_db": params[:, 0, :],
            "band0_cutoff_freq": params[:, 1, :],
            "band0_q_factor": params[:, 2, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
    ):
        bs = band0_cutoff_freq.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(band0_cutoff_freq)
        # ------------ band0 ------------
        b, a = biquad(
            band0_gain_db,  # gain_db
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "low_shelf",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Highshelf
# -----------------------------------------------------------------------------


class Highshelf(torch.nn.Module):
    """Highshelf filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_cutoff_freq: float = 200.0,
        max_cutoff_freq: float = 2000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band0_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 3

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor):
        param_dict = {
            "band0_gain_db": params[:, 0, :],
            "band0_cutoff_freq": params[:, 1, :],
            "band0_q_factor": params[:, 2, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
    ):
        bs = band0_cutoff_freq.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(band0_cutoff_freq)
        # ------------ band0 ------------
        b, a = biquad(
            band0_gain_db,  # gain_db
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "low_shelf",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Static FIR Filter
# -----------------------------------------------------------------------------


class StaticFIRFilter(torch.nn.Module):

    def __init__(
        self,
        sample_rate: float,
        n_taps: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        w0_initial: float = 30.0,
        pretrained: str = None,
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_taps = n_taps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.w0_initial = w0_initial
        self.pretrained = pretrained
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 0
        self.control_type = None

        self.net = SirenNet(1, hidden_dim, 1, num_layers, w0_initial=w0_initial)

        coords = torch.linspace(-1, 1, n_taps).view(1, 1, -1)
        self.register_buffer("coords", coords)

        if pretrained is not None:
            state_dict_path = pretrained
            print(f"\nLoading weights from {state_dict_path}")
            self.load_state_dict(torch.load(state_dict_path))

    def get_param_dict(self, params: torch.Tensor):
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train), self.get_param_dict(control_params)

    def extract_impulse_response(self, x: torch.Tensor):
        bs = x.size(0)
        impulse = self.net(self.coords.repeat(bs, 1, 1).permute(0, 2, 1))
        impulse = impulse.permute(0, 2, 1).squeeze(1)
        return impulse

    def process(self, x: torch.Tensor, train: bool = False):
        b = self.extract_impulse_response(x)
        output = lfilter_via_fsm(x, b)
        return output


# -----------------------------------------------------------------------------
# NONLINEARITIES
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Tanh
# -----------------------------------------------------------------------------


class TanhNonlinearity(torch.nn.Module):
    def __init__(self, sample_rate: float):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_control_params = 0
        self.control_type = None

    def get_param_dict(self, params: torch.Tensor):
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train)

    def process(self, x: torch.Tensor, train: bool = False):
        return torch.tanh(x), self.get_param_dict(control_params)


# -----------------------------------------------------------------------------
# Static MLP Nonlinearity
# -----------------------------------------------------------------------------


class StaticMLPNonlinearity(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        hidden_dim: int = 64,
        num_layers: int = 3,
        w0_initial: float = 30.0,
        pretrained: str = None,
        lr_multiplier=1.0,
    ):
        """
        Single learnable nonlinearity with no external conditioning parameters..

        Notes: For efficiecny, we could consider creating a lookup table for the
        output of the MLP and then use that to compute the output of the MLP.

        """
        super().__init__()
        self.sample_rate = sample_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.w0_initial = w0_initial
        self.pretrained = pretrained
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 0
        self.control_type = None

        self.net = SirenNet(1, hidden_dim, 1, num_layers, w0_initial=w0_initial)

        if pretrained is not None:
            state_dict_path = pretrained
            print(f"Loading weights from {state_dict_path}")
            self.load_state_dict(torch.load(state_dict_path))

    def get_param_dict(self, params: torch.Tensor):
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train), self.get_param_dict(control_params)

    def process(self, x: torch.Tensor, train: bool = False):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


# -----------------------------------------------------------------------------
# Static Rational Nonlinearity
# -----------------------------------------------------------------------------


class StaticRationalNonlinearity(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        degrees: Tuple[int, int] = (6, 5),
        init_approx_func: str = "tanh",
        lr_multiplier=1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.degrees = degrees
        self.init_approx_func = init_approx_func
        self.num_control_params = 0
        self.lr_multiplier = lr_multiplier
        self.control_type = None

        self.net = Rational(init_approx_func, degrees, version="A")

    def get_param_dict(self, params: torch.Tensor):
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train), self.get_param_dict(control_params)

    def process(self, x: torch.Tensor, train: bool = False):
        return self.net(x)
