import math
import torch
import torchaudio


# -----------------------------------------------------------------------------
# DSP Utils
# -----------------------------------------------------------------------------


def denormalize(norm_val, max_val, min_val):
    return (norm_val * (max_val - min_val)) + min_val


def denormalize_parameters(param_dict: dict, param_ranges: dict):
    """Given parameters on (0,1) restore them to the ranges expected by the DSP device."""
    denorm_param_dict = {}
    for param_name, param_val in param_dict.items():
        param_val_denorm = denormalize(
            param_val,
            param_ranges[param_name][1],
            param_ranges[param_name][0],
        )
        denorm_param_dict[param_name] = param_val_denorm
    return denorm_param_dict


def fft_freqz(b, a, n_fft: int = 512):
    """Compute the complex frequency response via FFT of an IIR filter.

    Args:
        b (torch.Tensor): Numerator coefficients with shape (bs, N)
        a (torch.Tensor): Denominator coefficients with shape (bs, N)
        n_fft (int): FFT size. Default: 512
    Returns:
        H (torch.Tensor): Complex frequency response with shape (bs, n_bins)
    """
    B = torch.fft.rfft(b, n_fft)
    A = torch.fft.rfft(a, n_fft)
    H = B / A
    return H


def fft_sosfreqz(sos: torch.Tensor, n_fft: int = 512):
    """Compute the complex frequency response via FFT of cascade of biquads

    Args:
        sos (torch.Tensor): Second order filter sections with shape (bs, n_sections, 6)
        n_fft (int): FFT size. Default: 512
    Returns:
        H (torch.Tensor): Overall complex frequency response with shape (bs, n_bins)
    """
    bs, n_sections, n_coeffs = sos.size()
    assert n_coeffs == 6  # must be second order
    for section_idx in range(n_sections):
        b = sos[:, section_idx, :3]
        a = sos[:, section_idx, 3:]
        if section_idx == 0:
            H = fft_freqz(b, a, n_fft=n_fft)
        else:
            H *= fft_freqz(b, a, n_fft=n_fft)
    return H


def freqdomain_fir(x, H, n_fft):
    X = torch.fft.rfft(x, n_fft)
    Y = X * H.type_as(X)
    y = torch.fft.irfft(Y, n_fft)
    return y


def lfilter_via_fsm(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor = None):
    """Use the frequency sampling method to approximate an IIR filter.
    The filter will be applied along the final dimension of x.
    Args:
        x (torch.Tensor): Time domain signal with shape (bs, ... , timesteps)
        b (torch.Tensor): Numerator coefficients with shape (bs, N).
        a (torch.Tensor): Denominator coefficients with shape (bs, N).
    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, ..., timesteps)
    """
    bs = x.size(0)

    # round up to nearest power of 2 for FFT
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # move coefficients to same device as x
    b = b.type_as(x)

    if a is None:
        # directly compute FFT of numerator coefficients
        H = torch.fft.rfft(b, n_fft)
    else:
        a = a.type_as(x)
        # compute complex response as ratio of polynomials
        H = fft_freqz(b, a, n_fft=n_fft)

    # add extra dims to broadcast filter across
    for _ in range(x.ndim - 2):
        H = H.unsqueeze(1)

    # apply as a FIR filter in the frequency domain
    y = freqdomain_fir(x, H, n_fft)

    # crop
    y = y[..., : x.shape[-1]]

    return y


def sosfilt_via_fsm(sos: torch.Tensor, x: torch.Tensor):
    """Use the frequency sampling method to approximate a cascade of second order IIR filters.

    The filter will be applied along the final dimension of x.
    Args:
        sos (torch.Tensor): Tensor of coefficients with shape (bs, n_sections, 6).
        x (torch.Tensor): Time domain signal with shape (bs, ... , timesteps)

    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, ..., timesteps)
    """
    bs = x.size(0)

    # round up to nearest power of 2 for FFT
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # compute complex response as ratio of polynomials
    H = fft_sosfreqz(sos, n_fft=n_fft)

    # add extra dims to broadcast filter across
    for _ in range(x.ndim - 2):
        H = H.unsqueeze(1)

    # apply as a FIR filter in the frequency domain
    y = freqdomain_fir(x, H, n_fft)

    # crop
    y = y[..., : x.shape[-1]]

    return y


def sosfilt(sos: torch.Tensor, x: torch.Tensor):
    """Apply cascade of second order IIR filters in the time domain.

    The filter will be applied along the final dimension of x.
    Args:
        sos (torch.Tensor): Tensor of coefficients with shape (bs, n_sections, 6).
        x (torch.Tensor): Time domain signal with shape (bs, chs, timesteps)

    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, chs, timesteps)

    Note: no gradient computation will occur in this function.

    """
    x_bs, chs, seq_len = x.size()
    sos_bs, n_sections, n_coeffs = sos.size()
    assert n_coeffs == 6  # must be second order
    assert chs == 1  # must be mono (for now)
    y = x.squeeze(1)
    with torch.no_grad():
        for section_idx in range(n_sections):
            b_coeffs = sos[:, section_idx, :3]
            a_coeffs = sos[:, section_idx, 3:]
            y = torchaudio.functional.lfilter(y, a_coeffs, b_coeffs)
    return y.unsqueeze(1)


def biquad(
    gain_db: torch.Tensor,
    cutoff_freq: torch.Tensor,
    q_factor: torch.Tensor,
    sample_rate: float,
    filter_type: str = "peaking",
):
    bs = gain_db.size(0)
    # reshape params
    gain_db = gain_db.view(bs, -1)
    cutoff_freq = cutoff_freq.view(bs, -1)
    q_factor = q_factor.view(bs, -1)

    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / sample_rate)
    alpha = torch.sin(w0) / (2 * q_factor)
    cos_w0 = torch.cos(w0)
    sqrt_A = torch.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)
    elif filter_type == "low_pass":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "high_pass":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = torch.stack([b0, b1, b2], dim=1).view(bs, -1)
    a = torch.stack([a0, a1, a2], dim=1).view(bs, -1)

    # normalize
    b = b.type_as(gain_db) / a0
    a = a.type_as(gain_db) / a0

    return b, a
