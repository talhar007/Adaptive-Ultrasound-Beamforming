"""Excitation pulse p0(t): a band-limited burst formed by convolving a
sinusoidal excitation with a Gaussian-windowed transducer impulse response.

This mirrors the pulse construction in the supplied notebook (section 3) and
is treated as fixed / non-learnable -- only its time-of-arrival (tau_tx) and
amplitude (w_tx) are learned by the model. (Learning the pulse shape itself
would be a natural future extension, but is out of scope here.)
"""
import numpy as np
import torch
import torch.nn.functional as F


def build_base_pulse(fc, fs, bw_frac=0.6, device='cpu'):
    """Returns a 1-D tensor: the broadband base pulse p0(t)."""
    t_cycle = torch.arange(0, 1 / fc, 1 / fs, device=device)
    excitation = torch.sin(2 * torch.pi * fc * t_cycle)

    t_ir = torch.arange(-3 / fc, 3 / fc, 1 / fs, device=device)
    sigma = np.sqrt(2 * np.log(2)) / (np.pi * fc * bw_frac)
    h_transducer = torch.exp(-0.5 * (t_ir / sigma) ** 2) * torch.sin(2 * torch.pi * fc * t_ir)
    h_transducer = h_transducer / torch.max(torch.abs(h_transducer))

    pulse = F.conv1d(
        excitation.view(1, 1, -1),
        h_transducer.view(1, 1, -1),
        padding=h_transducer.shape[-1] - 1,
    ).squeeze()
    return pulse
