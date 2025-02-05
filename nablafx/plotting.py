import io
import copy
import torch
import PIL.Image
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor

from nablafx.models import GreyBoxModel, BlackBoxModel
from nablafx.processors import (
    PhaseInversion,
    Gain,
    DCOffset,
    ParametricEQ,
    Lowpass,
    Highpass,
    StaticFIRFilter,
    StaticMLPNonlinearity,
    StaticRationalNonlinearity,
)


def fig2img(fig, dpi=120):
    """Convert a matplotlib figure to JPEG to be show in Tensorboard."""
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=dpi)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close("all")
    return image


def plot_static_params(param_dict: dict, param_ax: plt.Axes, batch_idx: int = 0):
    y_val = 0.90
    x_val = 0.025
    for param_idx, (param, param_val) in enumerate(param_dict.items()):
        param_ax.text(
            x_val,
            y_val,
            f"{param_val[batch_idx, ...].squeeze():>6.3f}",
            fontsize=9,
        )
        param_ax.text(
            x_val + 0.3,
            y_val,
            f"{param: <10s}",
            fontsize=9,
        )
        y_val -= 0.03
    param_ax.axis("off")


# -----------------------------------------------------------------------------
# Plot Phase Inversion Block
# -----------------------------------------------------------------------------


def plot_phase_inv(
    processor: PhaseInversion,
    param_dict: dict,
    input: torch.Tensor,
    batch_idx: int,
    response_ax: plt.Axes,
    param_ax: plt.Axes,
):
    w = np.linspace(1, 24000, num=50)
    h = np.ones(50) * -1

    response_ax.plot(w, h)
    response_ax.grid(c="lightgray")
    response_ax.set_xscale("log")
    response_ax.set_xlabel("Frequency (Hz)")
    response_ax.set_ylabel("Phase")
    response_ax.set_xlim([1, 24000])
    response_ax.set_ylim([-2, 2])
    response_ax.set_title("Phase Inversion")

    param_ax.axis("off")


# -----------------------------------------------------------------------------
# Plot Gain Block
# -----------------------------------------------------------------------------


def plot_gain(
    processor: Gain,
    param_dict: dict,
    input: torch.Tensor,
    batch_idx: int,
    response_ax: plt.Axes,
    param_ax: plt.Axes,
):
    min_gain_db = processor.param_ranges["gain_db"][0]
    max_gain_db = processor.param_ranges["gain_db"][1]
    seq_len = input.shape[-1]

    if processor.control_type in ["static", "static-cond"]:
        gain_db = param_dict["gain_db"]
        gain_db = gain_db[batch_idx].detach().cpu().numpy()

        w = np.linspace(1, 24000, num=50)
        h = np.ones(50) * gain_db

        response_ax.plot(w, h)
        response_ax.grid(c="lightgray")
        response_ax.set_xscale("log")
        response_ax.set_xlabel("Frequency (Hz)")
        response_ax.set_ylabel("Gain (dB)")
        response_ax.set_xlim([1, 24000])
        response_ax.set_ylim([min_gain_db - 2.0, max_gain_db + 2.0])
        response_ax.set_title("Gain")
        plot_static_params(param_dict, param_ax, batch_idx)

    elif processor.control_type in ["dynamic", "dynamic-cond"]:
        gain_db = param_dict["gain_db"]
        gain_db = gain_db[batch_idx].detach().cpu().numpy()
        input = input[batch_idx, 0, :].detach().cpu().numpy()

        w = np.linspace(0, seq_len - 1, num=10000, dtype=int)

        h1 = gain_db[w]
        h2 = input

        response_ax2 = response_ax.twinx()
        response_ax.set_zorder(response_ax2.get_zorder() + 1)
        response_ax.patch.set_visible(False)

        response_ax.plot(w, h1)
        response_ax2.plot(h2, c="lightgray", alpha=0.5)

        response_ax.grid(c="lightgray")
        response_ax.set_xlabel("Samples")
        response_ax.set_ylabel("Gain (dB)")
        response_ax.set_xlim([0, seq_len])
        response_ax.set_ylim([min_gain_db - 2.0, max_gain_db + 2.0])
        response_ax.set_title("Gain")

        response_ax2.set_ylabel("Input")
        response_ax2.set_ylim([-1.1, 1.1])

        param_ax.axis("off")


# -----------------------------------------------------------------------------
# Plot DC Offset Block
# -----------------------------------------------------------------------------


def plot_dc_offset(
    processor: Gain,
    param_dict: dict,
    input: torch.Tensor,
    batch_idx: int,
    response_ax: plt.Axes,
    param_ax: plt.Axes,
):
    min_offset = processor.param_ranges["offset"][0]
    max_offset = processor.param_ranges["offset"][1]
    seq_len = input.shape[-1]

    if processor.control_type in ["static", "static-cond"]:
        offset = param_dict["offset"]
        offset = offset[batch_idx].detach().cpu().numpy()

        w = np.linspace(1, 24000, num=50)
        h = np.ones(50) * offset

        response_ax.plot(w, h)
        response_ax.grid(c="lightgray")
        response_ax.set_xscale("log")
        response_ax.set_xlabel("Frequency (Hz)")
        response_ax.set_ylabel("Offset")
        response_ax.set_xlim([1, 24000])
        response_ax.set_ylim([min_offset - 0.5, max_offset + 0.5])
        response_ax.set_title("DC Offset")
        plot_static_params(param_dict, param_ax, batch_idx)

    elif processor.control_type in ["dynamic", "dynamic-cond"]:
        offset = param_dict["offset"]
        offset = offset[batch_idx].detach().cpu().numpy()
        input = input[batch_idx, 0, :].detach().cpu().numpy()

        w = np.linspace(0, seq_len - 1, num=10000, dtype=int)
        h1 = offset[w]
        h2 = input

        response_ax2 = response_ax.twinx()
        response_ax.set_zorder(response_ax2.get_zorder() + 1)
        response_ax.patch.set_visible(False)

        response_ax.plot(w, h1)
        response_ax2.plot(h2, c="lightgray", alpha=0.5)

        response_ax.grid(c="lightgray")
        response_ax.set_xlabel("Samples")
        response_ax.set_ylabel("Offset")
        response_ax.set_xlim([0, seq_len])
        response_ax.set_ylim([min_offset - 0.5, max_offset + 0.5])
        response_ax.set_title("DC Offset")

        response_ax2.set_ylabel("Input")
        response_ax2.set_ylim([-1.1, 1.1])

        param_ax.axis("off")


# -----------------------------------------------------------------------------
# Plot Parametric EQ Block
# -----------------------------------------------------------------------------


def plot_parametric_eq(
    processor: ParametricEQ,
    param_dict: dict,
    input: torch.Tensor,
    batch_idx: int,
    response_ax: plt.Axes,
    param_ax: plt.Axes,
):
    min_gain_db = processor.min_gain_db
    max_gain_db = processor.max_gain_db
    seq_len = input.shape[-1]
    sample_rate = processor.sample_rate

    if processor.control_type in ["static", "static-cond"]:
        # get filter coefficients
        sos = processor.compute_coefficients(sample_rate, **param_dict)
        sos = sos[batch_idx, :].detach().cpu().numpy()

        # measure frequency response
        w, h = scipy.signal.sosfreqz(sos, worN=65536, fs=sample_rate)
        response_ax.plot(w, 20 * np.log10(np.abs(h) + 1e-8))
        response_ax.grid(c="lightgray")
        response_ax.set_xscale("log")
        response_ax.set_xlabel("Frequency (Hz)")
        response_ax.set_ylabel("Gain (dB)")
        response_ax.set_xlim([1, sample_rate / 2])
        response_ax.set_ylim([min_gain_db - 24.0, max_gain_db + 24.0])
        response_ax.set_title("Parametric EQ")
        plot_static_params(param_dict, param_ax, batch_idx)

    elif processor.control_type in ["dynamic", "dynamic-cond"]:

        # plot frequency response for each block
        num_blocks = param_dict["low_shelf_gain_db"].shape[-1]
        for i in range(num_blocks):
            # get param_dict for block i
            param_dict_block = {}
            for k, v in param_dict.items():
                param_dict_block[k] = v[:, i].unsqueeze(-1)

            # get filter coefficients for block i
            sos = processor.compute_coefficients(sample_rate, **param_dict_block)
            sos = sos[batch_idx, :].detach().cpu().numpy()

            # measure frequency response
            w, h = scipy.signal.sosfreqz(sos, worN=65536, fs=sample_rate)
            response_ax.plot(w, 20 * np.log10(np.abs(h) + 1e-8), c="lightgray")

        response_ax.grid(c="lightgray")
        response_ax.set_xscale("log")
        response_ax.set_xlabel("Frequency (Hz)")
        response_ax.set_ylabel("Gain (dB)")
        response_ax.set_xlim([1, sample_rate / 2])
        response_ax.set_ylim([min_gain_db - 24.0, max_gain_db + 24.0])
        response_ax.set_title("Parametric EQ")

        # plot parameters values
        param_ax2 = param_ax.twinx()
        param_ax.set_zorder(param_ax2.get_zorder() + 1)
        param_ax.patch.set_visible(False)

        w = np.linspace(0, seq_len - 1, num=num_blocks, dtype=int)
        h2 = input[batch_idx, 0, :].detach().cpu().numpy()
        param_ax2.plot(h2, c="lightgray", alpha=0.5)

        for k, v in param_dict.items():
            param_val = v[batch_idx].detach().cpu().numpy()

            h1 = param_val

            param_ax.plot(w, h1, label=k, alpha=0.5)
            param_ax.grid(c="lightgray")
            param_ax.set_xlabel("Samples")
            param_ax.set_ylabel("Param Value")

        param_ax.legend()


# -----------------------------------------------------------------------------
# Plot Lowpass Block
# -----------------------------------------------------------------------------


def plot_lowpass(
    processor: Lowpass,
    param_dict: dict,
    input: torch.Tensor,
    batch_idx: int,
    response_ax: plt.Axes,
    param_ax: plt.Axes,
):
    seq_len = input.shape[-1]
    sample_rate = processor.sample_rate

    if processor.control_type in ["static", "static-cond"]:
        # get filter coefficients
        sos = processor.compute_coefficients(sample_rate, **param_dict)
        sos = sos[batch_idx, :].detach().cpu().numpy()

        # measure frequency response
        w, h = scipy.signal.sosfreqz(sos, worN=65536, fs=sample_rate)
        response_ax.plot(w, 20 * np.log10(np.abs(h) + 1e-8))
        response_ax.grid(c="lightgray")
        response_ax.set_xscale("log")
        response_ax.set_xlabel("Frequency (Hz)")
        response_ax.set_ylabel("Gain (dB)")
        response_ax.set_xlim([10, sample_rate / 2])
        response_ax.set_xlim([1, sample_rate / 2])
        response_ax.set_ylim([-40.0, 12.0])
        response_ax.set_title("Lowpass")
        plot_static_params(param_dict, param_ax, batch_idx)

    elif processor.control_type in ["dynamic", "dynamic-cond"]:

        # plot frequency response for each block
        num_blocks = param_dict["cutoff_freq"].shape[-1]
        for i in range(num_blocks):
            # get param_dict for block i
            param_dict_block = {}
            for k, v in param_dict.items():
                param_dict_block[k] = v[:, i].unsqueeze(-1)

            # get filter coefficients for block i
            sos = processor.compute_coefficients(sample_rate, **param_dict_block)
            sos = sos[batch_idx, :].detach().cpu().numpy()

            # measure frequency response
            w, h = scipy.signal.sosfreqz(sos, worN=65536, fs=sample_rate)
            response_ax.plot(w, 20 * np.log10(np.abs(h) + 1e-8), c="lightgray")

        response_ax.grid(c="lightgray")
        response_ax.set_xscale("log")
        response_ax.set_xlabel("Frequency (Hz)")
        response_ax.set_ylabel("Gain (dB)")
        response_ax.set_xlim([1, sample_rate / 2])
        response_ax.set_ylim([-40.0, 12.0])
        response_ax.set_title("Lowpass")

        # plot parameters values
        param_ax2 = param_ax.twinx()
        param_ax.set_zorder(param_ax2.get_zorder() + 1)
        param_ax.patch.set_visible(False)

        w = np.linspace(0, seq_len - 1, num=num_blocks, dtype=int)
        h2 = input[batch_idx, 0, :].detach().cpu().numpy()
        param_ax2.plot(h2, c="lightgray", alpha=0.5)

        for k, v in param_dict.items():
            param_val = v[batch_idx].detach().cpu().numpy()

            h1 = param_val

            param_ax.plot(w, h1, label=k, alpha=0.5)
            param_ax.grid(c="lightgray")
            param_ax.set_xlabel("Samples")
            param_ax.set_ylabel("Param Value")

        param_ax.legend()


# -----------------------------------------------------------------------------
# Plot Highpass Block
# -----------------------------------------------------------------------------


def plot_highpass(
    processor: Highpass,
    param_dict: dict,
    input: torch.Tensor,
    batch_idx: int,
    response_ax: plt.Axes,
    param_ax: plt.Axes,
):
    seq_len = input.shape[-1]
    sample_rate = processor.sample_rate

    if processor.control_type in ["static", "static-cond"]:
        # get filter coefficients
        sos = processor.compute_coefficients(sample_rate, **param_dict)
        sos = sos[batch_idx, :].detach().cpu().numpy()

        # measure frequency response
        w, h = scipy.signal.sosfreqz(sos, worN=65536, fs=sample_rate)
        response_ax.plot(w, 20 * np.log10(np.abs(h) + 1e-8))
        response_ax.grid(c="lightgray")
        response_ax.set_xscale("log")
        response_ax.set_xlabel("Frequency (Hz)")
        response_ax.set_ylabel("Gain (dB)")
        response_ax.set_xlim([1, sample_rate / 2])
        response_ax.set_ylim([-40.0, 12.0])
        response_ax.set_title("Highpass")
        plot_static_params(param_dict, param_ax, batch_idx)

    elif processor.control_type in ["dynamic", "dynamic-cond"]:

        # plot frequency response for each block
        num_blocks = param_dict["cutoff_freq"].shape[-1]
        for i in range(num_blocks):
            # get param_dict for block i
            param_dict_block = {}
            for k, v in param_dict.items():
                param_dict_block[k] = v[:, i].unsqueeze(-1)

            # get filter coefficients for block i
            sos = processor.compute_coefficients(sample_rate, **param_dict_block)
            sos = sos[batch_idx, :].detach().cpu().numpy()

            # measure frequency response
            w, h = scipy.signal.sosfreqz(sos, worN=65536, fs=sample_rate)
            response_ax.plot(w, 20 * np.log10(np.abs(h) + 1e-8), c="lightgray")

        response_ax.grid(c="lightgray")
        response_ax.set_xscale("log")
        response_ax.set_xlabel("Frequency (Hz)")
        response_ax.set_ylabel("Gain (dB)")
        response_ax.set_xlim([1, sample_rate / 2])
        response_ax.set_ylim([-40.0, 12.0])
        response_ax.set_title("Highpass")

        # plot parameters values
        param_ax2 = param_ax.twinx()
        param_ax.set_zorder(param_ax2.get_zorder() + 1)
        param_ax.patch.set_visible(False)

        w = np.linspace(0, seq_len - 1, num=num_blocks, dtype=int)
        h2 = input[batch_idx, 0, :].detach().cpu().numpy()
        param_ax2.plot(h2, c="lightgray", alpha=0.5)

        for k, v in param_dict.items():
            param_val = v[batch_idx].detach().cpu().numpy()

            h1 = param_val

            param_ax.plot(w, h1, label=k, alpha=0.5)
            param_ax.grid(c="lightgray")
            param_ax.set_xlabel("Samples")
            param_ax.set_ylabel("Param Value")

        param_ax.legend()


# -----------------------------------------------------------------------------
# Plot FIR Filter Block
# -----------------------------------------------------------------------------


def plot_fir_filter(
    processor: StaticFIRFilter,
    param_dict: dict,
    input: torch.Tensor,
    batch_idx: int,
    response_ax: plt.Axes,
    param_ax: plt.Axes,
):
    seq_len = input.shape[-1]
    sample_rate = processor.sample_rate
    # reconstruct param tensor from dict
    params = torch.zeros(1, len(list(param_dict.keys())))
    for idx, (param_name, param_val) in enumerate(param_dict.items()):
        params[:, idx] = param_val[batch_idx : batch_idx + 1, :]

    params = params.view(1, -1)
    device = next(processor.parameters()).device
    num_seconds = processor.n_taps / processor.sample_rate
    t = torch.linspace(0, num_seconds, processor.n_taps)

    params = params.to(device)

    with torch.no_grad():
        impulse = processor.extract_impulse_response(input[batch_idx, :, :])
    impulse = impulse.cpu().squeeze().numpy()

    # impulse response
    response_ax.plot(t, impulse)
    response_ax.grid(c="lightgray")
    response_ax.set_xlabel("Time (s)")
    response_ax.set_ylabel("Amplitude")
    response_ax.set_xlim([0, num_seconds])
    # axs[0, 2].set_ylim([-42.0, 42.0])
    response_ax.set_title("FIR Filter")

    # frequency response
    w, h = scipy.signal.freqz(impulse, worN=65536, fs=sample_rate)
    param_ax.plot(w, 20 * np.log10(np.abs(h) + 1e-8))
    param_ax.grid(c="lightgray")
    param_ax.set_xscale("log")
    param_ax.set_xlabel("Frequency (Hz)")
    param_ax.set_ylabel("Gain (dB)")
    param_ax.set_xlim([1, sample_rate / 2])
    # param_ax.set_ylim([-40.0, 12.0])

    # plot_static_params(param_dict, param_ax, batch_idx)


# -----------------------------------------------------------------------------
# Plot Static MLP Nonlinearity Block
# -----------------------------------------------------------------------------


def plot_static_mlp_nonlinearity(
    processor: StaticMLPNonlinearity,
    param_dict: dict,
    input: torch.Tensor,
    batch_idx: int,
    response_ax: plt.Axes,
    param_ax: plt.Axes,
):
    # reconstruct param tensor from dict
    params = torch.zeros(1, len(list(param_dict.keys())))
    for idx, (param_name, param_val) in enumerate(param_dict.items()):
        params[:, idx] = param_val[batch_idx : batch_idx + 1, :]

    # params are stored directory in NL blocks
    x = torch.linspace(-4.0, 4.0, steps=100).view(1, 1, -1)
    params = params.view(1, -1)

    y_target = torch.tanh(x)

    device = next(processor.parameters()).device
    params = params.to(device)
    x = x.to(device)

    with torch.no_grad():
        y, _ = processor(x, None)

    y = y.cpu()
    x = x.cpu()

    response_ax.plot(x.squeeze(), y_target.squeeze(), c="lightgray")
    response_ax.plot(x.squeeze(), y.squeeze())
    # axs[1].set_aspect("equal", adjustable="box")
    response_ax.set_xlabel("Input")
    response_ax.set_ylabel("Output")
    response_ax.set_title("Nonlinearity")
    response_ax.grid(c="lightgray")

    plot_static_params(param_dict, param_ax, batch_idx)
    # param_ax.axis("off")


# -----------------------------------------------------------------------------
# Plot Static Rational Nonlinearity Block
# -----------------------------------------------------------------------------


def plot_static_rational_nonlinearity(
    processor: StaticRationalNonlinearity,
    param_dict: dict,
    input: torch.Tensor,
    batch_idx: int,
    response_ax: plt.Axes,
    param_ax: plt.Axes,
):
    # reconstruct param tensor from dict
    params = torch.zeros(1, len(list(param_dict.keys())))
    for idx, (param_name, param_val) in enumerate(param_dict.items()):
        params[:, idx] = param_val[batch_idx : batch_idx + 1, :]

    # params are stored directory in NL blocks
    x = torch.linspace(-4.0, 4.0, steps=100).view(1, 1, -1)
    params = params.view(1, -1)

    y_target = torch.tanh(x)

    device = next(processor.parameters()).device
    params = params.to(device)
    x = x.to(device)

    with torch.no_grad():
        y, _ = processor(x, None)

    y = y.cpu()
    x = x.cpu()

    response_ax.plot(x.squeeze(), y_target.squeeze(), c="lightgray")
    response_ax.plot(x.squeeze(), y.squeeze())
    # axs[1].set_aspect("equal", adjustable="box")
    response_ax.set_xlabel("Input")
    response_ax.set_ylabel("Output")
    response_ax.set_title("Nonlinearity")
    response_ax.grid(c="lightgray")

    plot_static_params(param_dict, param_ax, batch_idx)
    # param_ax.axis("off")


# -----------------------------------------------------------------------------
# Plot Gray-box Model Block by Block
# -----------------------------------------------------------------------------


def plot_gb_model(
    model: GreyBoxModel,
    param_dict_list: list,
    input: torch.Tensor,
    batch_idx: int = 0,
    filename: str = None,
):
    num_blocks = len(model.processor.processors)
    fig, axs = plt.subplots(
        nrows=num_blocks,
        ncols=2,
        figsize=(16, 6 * num_blocks),
        # gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    for i, (prc, param_dict) in enumerate(zip(model.processor.processors, param_dict_list)):
        if isinstance(prc, PhaseInversion):
            plot_phase_inv(
                processor=prc,
                param_dict=param_dict,
                input=input,
                batch_idx=batch_idx,
                response_ax=axs[i, 0] if num_blocks > 1 else axs[0],
                param_ax=axs[i, 1] if num_blocks > 1 else axs[1],
            )
        elif isinstance(prc, Gain):
            plot_gain(
                processor=prc,
                param_dict=param_dict,
                input=input,
                batch_idx=batch_idx,
                response_ax=axs[i, 0] if num_blocks > 1 else axs[0],
                param_ax=axs[i, 1] if num_blocks > 1 else axs[1],
            )
        elif isinstance(prc, DCOffset):
            plot_dc_offset(
                processor=prc,
                param_dict=param_dict,
                input=input,
                batch_idx=batch_idx,
                response_ax=axs[i, 0] if num_blocks > 1 else axs[0],
                param_ax=axs[i, 1] if num_blocks > 1 else axs[1],
            )
        elif isinstance(prc, ParametricEQ):
            plot_parametric_eq(
                processor=prc,
                param_dict=param_dict,
                input=input,
                batch_idx=batch_idx,
                response_ax=axs[i, 0] if num_blocks > 1 else axs[0],
                param_ax=axs[i, 1] if num_blocks > 1 else axs[1],
            )
        elif isinstance(prc, Lowpass):
            plot_lowpass(
                processor=prc,
                param_dict=param_dict,
                input=input,
                batch_idx=batch_idx,
                response_ax=axs[i, 0] if num_blocks > 1 else axs[0],
                param_ax=axs[i, 1] if num_blocks > 1 else axs[1],
            )
        elif isinstance(prc, Highpass):
            plot_highpass(
                processor=prc,
                param_dict=param_dict,
                input=input,
                batch_idx=batch_idx,
                response_ax=axs[i, 0] if num_blocks > 1 else axs[0],
                param_ax=axs[i, 1] if num_blocks > 1 else axs[1],
            )
        elif isinstance(prc, StaticMLPNonlinearity):
            plot_static_mlp_nonlinearity(
                processor=prc,
                param_dict=param_dict,
                input=input,
                batch_idx=batch_idx,
                response_ax=axs[i, 0] if num_blocks > 1 else axs[0],
                param_ax=axs[i, 1] if num_blocks > 1 else axs[1],
            )
        elif isinstance(prc, StaticRationalNonlinearity):
            plot_static_rational_nonlinearity(
                processor=prc,
                param_dict=param_dict,
                input=input,
                batch_idx=batch_idx,
                response_ax=axs[i, 0] if num_blocks > 1 else axs[0],
                param_ax=axs[i, 1] if num_blocks > 1 else axs[1],
            )
        elif isinstance(prc, StaticFIRFilter):
            plot_fir_filter(
                processor=prc,
                param_dict=param_dict,
                input=input,
                batch_idx=batch_idx,
                response_ax=axs[i, 0] if num_blocks > 1 else axs[0],
                param_ax=axs[i, 1] if num_blocks > 1 else axs[1],
            )

    plt.tight_layout(h_pad=0.5)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    return fig2img(fig)


# -----------------------------------------------------------------------------
# Plot Frequency/Phase Response (measured in steps)
# -----------------------------------------------------------------------------


def plot_frequency_response_steps(
    model: [GreyBoxModel, BlackBoxModel],
    filename: str = None,
):
    # prepare sinusoids
    f1 = 10  # start frequency
    f2 = 24000  # stop frequency
    num_f = 200  # number of frequencies
    fs = 96000  # sample rate
    T = 5  # duration in seconds
    amps = [1.0, 0.1, 0.01, 0.001, 0.0001]  # amplitudes
    batch_size = 5
    num_batches = int(np.ceil(num_f / batch_size))
    t = np.arange(0, np.round(fs * T - 1) / fs, 1 / fs)  # time axis
    f = np.logspace(np.log10(f1), np.log10(f2), num=num_f)  # frequency axis

    x_in = torch.zeros((num_f, t.size))
    for i in range(num_f):
        x_in[i] = torch.from_numpy(np.sin(2 * np.pi * f[i] * t)).float()

    # set controls
    if model.num_controls > 0:
        controls = 0.5 * torch.ones((batch_size, model.num_controls))
        controls = controls.cuda()
    else:
        controls = None

    # prepare plot
    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(16, 12),
    )

    x_in_min = amps[-1] * x_in.numpy()
    model.eval()

    # measure response at different amplitudes
    for i, a in enumerate(amps):
        print(f"Amplitude: {a}")

        # prepare input and output
        x = a * x_in
        y = torch.zeros_like(x)

        # run the model
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        x = x.cuda()
        y = y.cuda()
        for i in range(num_batches):
            model.reset_states()
            x_batch = x[i * batch_size : (i + 1) * batch_size, ...]
            with torch.no_grad():
                y_batch = model(x_batch, controls)
            y[i * batch_size : (i + 1) * batch_size, ...] = y_batch
            del x_batch, y_batch

        x = x.squeeze(1).detach().cpu().numpy()
        y = y.squeeze(1).detach().cpu().numpy()

        # take last samples to make sure we have steady state
        x = x[:, -T * int(fs / f1) :]
        y = y[:, -T * int(fs / f1) :]

        # remove DC offset
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)

        # magnitude
        # mag = np.abs(y).max(axis=-1) / np.abs(x).max(axis=-1)

        # magnitude (relative to minimum input)
        mag = np.abs(y).max(axis=-1) / np.abs(x_in_min).max(axis=-1)

        # phase difference between input and output
        z = x * y
        z = z.sum(axis=-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1))
        ph = np.arccos(np.clip(z, -1, 1)) * 360 / np.pi - 180

        # plot
        axs[0].plot(f, 20 * np.log10(mag + 1e-8), label=f"{20*np.log10(a):.0f}dB")
        axs[1].plot(f, ph, label=f"{a:.5f}")

        del x, y, mag, ph, z

    axs[0].set_title("Frequency Response (Relative to Minimum Input)")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Gain (dB)")
    axs[0].set_xlim([20, 20000])
    # axs[0].set_ylim([-80, 60])
    axs[0].grid(c="lightgray")
    axs[0].set_xscale("log")
    axs[0].legend()

    axs[1].set_title("Phase Response")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Phase (deg)")
    axs[1].set_xlim([20, 20000])
    axs[1].set_ylim([-200, 200])
    axs[1].grid(c="lightgray")
    axs[1].set_xscale("log")

    plt.tight_layout(h_pad=0.5)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    return fig2img(fig)


# -----------------------------------------------------------------------------
# Plot Frequency/Phase Response (measured with exponential sine sweep)
# -----------------------------------------------------------------------------


def plot_frequency_response(
    model: [GreyBoxModel, BlackBoxModel],
    filename: str = None,
):
    # prepare sweep
    f1 = 10  # start frequency
    f2 = 48000  # stop frequency
    fs = 192000  # sample rate
    T = 10  # duration in seconds
    L = T / np.log(f2 / f1)  # constant used in sweep calculation
    t = np.arange(0, np.round(fs * T - 1) / fs, 1 / fs)  # time axis
    x = np.sin(2 * np.pi * f1 * L * np.exp(t / L))
    taper = scipy.signal.tukey(x.size, 0.2)  # fade in/out
    x = np.float32(x * taper)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

    # set controls
    if model.num_controls > 0:
        controls = 0.5 * torch.ones((1, model.num_controls))
        controls = controls.cuda()
    else:
        controls = None

    # prepare input
    x = x.repeat(6, 1, 1)
    c = torch.tensor([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]).view(-1, 1, 1)  # 0db -> -100db
    x = x * c

    # run the model
    x = x.cuda()
    y = model(x, controls)

    x = x.squeeze(1).detach().cpu().numpy()
    y = y.squeeze(1).detach().cpu().numpy()

    x = np.float64(x)
    y = np.float64(y)

    # measure frequency response
    X = scipy.fft.fft(x, axis=-1)
    Y = scipy.fft.fft(y, axis=-1)
    H = Y / X
    H = H[0 : int(H.size / 2 + 1)]
    print(H.shape)
    mag = np.abs(H)
    ph = np.angle(H, deg=True)
    f = np.linspace(0, fs / 2, H.shape[-1])

    fig, axs = plt.subplots(
        nrows=2 * H.shape[0],
        ncols=1,
        figsize=(16, 12 * H.shape[0]),
        # gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    # plot
    labels = ["0db", "-20db", "-40db", "-60db", "-80db", "-100db"]
    for i in range(H.shape[0]):
        axs[2 * i].plot(f, 20 * np.log10(mag[i] + 1e-8))
        axs[2 * i].set_xlabel("Frequency (Hz)")
        axs[2 * i].set_ylabel("Gain (dB)")
        axs[2 * i].set_xlim([20, 20000])
        axs[2 * i].set_ylim([-80, 60])
        axs[2 * i].set_title(f"Frequency Response ({labels[i]})")
        axs[2 * i].grid(c="lightgray")
        axs[2 * i].set_xscale("log")

        axs[2 * i + 1].plot(f, ph[i])
        axs[2 * i + 1].set_xlabel("Frequency (Hz)")
        axs[2 * i + 1].set_ylabel("Phase (deg)")
        axs[2 * i + 1].set_xlim([20, 20000])
        axs[2 * i + 1].set_ylim([-200, 200])
        axs[2 * i + 1].set_title(f"Phase Response ({labels[i]})")
        axs[2 * i + 1].grid(c="lightgray")
        axs[2 * i + 1].set_xscale("log")

    # axs[0].plot(f, 20 * np.log10(mag + 1e-8))
    # axs[0].set_xlabel("Frequency (Hz)")
    # axs[0].set_ylabel("Gain (dB)")
    # axs[0].set_xlim([20, 20000])
    # axs[0].set_ylim([-80, 60])

    # axs[1].plot(f, ph)
    # axs[1].set_xlabel("Frequency (Hz)")
    # axs[1].set_ylabel("Phase (deg)")
    # axs[1].set_xlim([20, 20000])

    # axs[0].set_title("Frequency Response (0db)")
    # axs[1].set_title("Phase Response")
    # axs[0].grid(c="lightgray")
    # axs[1].grid(c="lightgray")
    # axs[0].set_xscale("log")
    # axs[1].set_xscale("log")

    plt.tight_layout(h_pad=0.5)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    return fig2img(fig)
