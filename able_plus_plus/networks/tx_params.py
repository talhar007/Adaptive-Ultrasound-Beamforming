"""Learnable global per-element transmit apodization and firing delays.

Extends the paper (which learns only receive apodization): the transmit
side of the ACQUISITION is also learned, jointly with the receive MLP, by
the same optimizer against the same loss —

    {MLP theta, w_tx, tau_tx}  <-  argmin  L_total(reconstruction, GT)

The parameters are applied on the SIMULATOR side (ForwardModel.simulate_tx),
i.e. before transmission — element m fires with amplitude w_tx[m] at
firing delay tau_tx[m] — the physically correct place: real transmit
apodization and timing shape the wave as it leaves the array, not the
receive processing.

Parameterization: ONE global vector per element (no depth dependence),
2 * M parameters total, both bounded through tanh:

    w_tx   = tanh(w_raw)              in (-1, 1)
    tau_tx = max_delay * tanh(t_raw)  in (-max_delay, +max_delay) samples

Two properties fixed from the first version, both about giving the
optimizer real room to move:

1. RANDOM per-element initialization (not a shared constant). Previously
   every element started at the SAME w_raw / tau_raw value (uniform
   firing) — after training, w_tx had barely moved from that shared
   starting point (final range ~0.92-0.99 for an init of ~0.96), which
   reads as "the network didn't learn anything." Two causes: (a) tanh
   saturates, so a large shared init sits on a near-flat part of the
   curve where d(tanh)/dx is small everywhere at once; (b) with every
   element starting identical, the optimizer has no reason to break the
   symmetry unless the loss gradient does so element-by-element, which a
   spatially-averaged image loss does only weakly. Independent random
   initialization (w_raw, tau_raw ~ N(0, std)) starts every element at a
   DIFFERENT point on the tanh curve and breaks the symmetry from step
   one, so the learned end state is visibly distinct from any "uniform"
   starting condition, whichever direction training pushes it.

2. The delay bound is expressed in PERIODS OF THE CARRIER SINUSOID, not
   raw samples: max_delay = max_delay_periods * (fs / fc). This ties the
   bound to the physical unit that actually matters for a fractional-delay
   shift — the pulse's own oscillation period (fs/fc = 8 samples at the
   default fc=5 MHz / fs=40 MHz) — rather than an arbitrary sample count.
   Default is 1.0 period (wider than the earlier fixed +/-4-sample bound,
   i.e. +/-0.5 period) while staying inside the base pulse's energetic
   region: build_base_pulse's Gaussian envelope has sigma ~= 0.6 period, so
   +/-1 period is ~1.6 sigma — still well within the pulse's support,
   short of its vanishing tail at +/-3 periods where the delay gradient
   (the local slope of the echo) would go to zero.
"""
import torch
import torch.nn as nn


class TxParams(nn.Module):
    def __init__(self, M, fc=5e6, fs=40e6, max_delay_periods=1.0,
                 w_init_std=1.5, tau_init_std=1.0, seed=None):
        super().__init__()
        self.M = M
        self.period_samples = fs / fc
        self.max_delay = max_delay_periods * self.period_samples

        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        # Independent random draw per element -- see module docstring point 1.
        self.w_raw   = nn.Parameter(torch.randn(M, generator=gen) * w_init_std)
        self.tau_raw = nn.Parameter(torch.randn(M, generator=gen) * tau_init_std)

    def forward(self):
        """returns: w_tx [M] in (-1, 1),  tau_tx [M] in (-max_delay, +max_delay)."""
        return torch.tanh(self.w_raw), self.max_delay * torch.tanh(self.tau_raw)
