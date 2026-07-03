"""Differentiable forward acoustic model + DAS adjoint.

**Architecture after professor meeting (see professor_meeting_context.md):**

The forward model is a FIXED data generator only — no parameters are
learned through it and no gradients flow into it.  All geometry is
pre-computed once in __init__ as non-differentiable buffers.

Pipeline:
    scatterer_map [B, nx*nz]
        -> simulate()           (forward model,  under no_grad at call site)
    rf_data [B, M, N_t]
        -> das_adjoint()        (delay-and-align, returns two tensors)
    pre_summed_samples [B, M*M, nx*nz]   <-- MLP input (the bridge)
    das_image [B, nx*nz]                 <-- DAS baseline (uniform weights)

The MLP replaces the trivial "sum everything with weight=1" of DAS with
learned per-pixel apodization weights, operating entirely on
pre_summed_samples.  Backprop never leaves the MLP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import build_grids, linear_array, element_pixel_distances
from .pulse import build_base_pulse


class ForwardModel(nn.Module):
    """Fixed (non-trainable) ultrasound physics engine.

    All geometry — element-to-pixel distances, propagation delays in
    samples, linear-interpolation indices and gain-weighted coefficients —
    is computed once during __init__ and registered as buffers.
    The model has NO nn.Parameters; it is a frozen physics simulator.
    """

    def __init__(self, M=64, pitch=1e-3, c=5920.0, fc=5e6, fs=40e6,
                 nx=128, nz=128, x_lim=32e-3, z_start=10e-3, z_end=74e-3,
                 device='cpu'):
        super().__init__()
        self.M, self.c, self.fc, self.fs = M, c, fc, fs
        self.nx, self.nz = nx, nz

        x_vec, z_vec, grid_X, grid_Z = build_grids(x_lim, z_start, z_end, nx, nz, device)
        elem_x = linear_array(M, pitch, device)   # [M, 1]

        # ----------------------------------------------------------------
        # Pre-compute ALL geometry under no_grad — these never change
        # ----------------------------------------------------------------
        with torch.no_grad():
            dist = element_pixel_distances(elem_x, grid_X, grid_Z)   # [M, P]

            # Notebook convention: dim0 = rx element, dim1 = tx element
            #   dist_rx.unsqueeze(1) -> [M, 1, P]   (rx varies along dim0)
            #   dist_tx.unsqueeze(0) -> [1, M, P]   (tx varies along dim1)
            #   total_dist[rx, tx, pixel] = dist_rx[rx] + dist_tx[tx]
            dist_rx = dist.unsqueeze(1)                               # [M, 1, P]
            dist_tx = dist.unsqueeze(0)                               # [1, M, P]
            total_dist = dist_tx + dist_rx                            # [M, M, P]
            total_gain = 1.0 / (total_dist + 1e-3)                   # 1/r amplitude decay

            geo_delay = (total_dist / c) * fs                        # [M, M, P] in samples
            geo_flat  = geo_delay.view(M * M, -1)                    # [M*M, P]
            gain_flat = total_gain.view(M * M, -1)                   # [M*M, P]

            max_delay = geo_flat.max().item()
            pulse_pad = int(fs / fc * 10)
            buf = int(max_delay + pulse_pad + 50)
            self.buffer_len = buf
            self.N_t = buf

            # Linear interpolation indices (integer — no grad needed)
            idx_f = geo_flat.floor().long()
            idx_c = idx_f + 1
            frac  = geo_flat - idx_f.float()

            valid = (idx_c < buf)
            # Weights absorb both the fractional-delay coefficient AND 1/r gain
            w_f = (1.0 - frac) * gain_flat * valid.float()
            w_c =        frac  * gain_flat * valid.float()

            self.register_buffer('idx_floor', idx_f.clamp(0, buf - 1))
            self.register_buffer('idx_ceil',  idx_c.clamp(0, buf - 1))
            self.register_buffer('w_floor',   w_f)
            self.register_buffer('w_ceil',    w_c)

        self.register_buffer('base_pulse',
                             build_base_pulse(fc, fs, device=device))

    # ------------------------------------------------------------------
    # 1. Forward model  (physics simulation, fixed)
    # ------------------------------------------------------------------
    def simulate(self, scatterer_maps):
        """scatterer_maps: [B, nx*nz]   ->   rf_data: [B, M_rx, N_t]

        Spreads each scatterer's amplitude to the correct time samples for
        every (tx, rx) element pair, convolves with the broadband pulse,
        then sums the M simultaneously-fired transmit contributions to
        produce the signal received at each of the M receive channels.
        """
        B  = scatterer_maps.shape[0]
        MM = self.M * self.M

        spike = scatterer_maps.new_zeros(B, MM, self.buffer_len)
        img   = scatterer_maps.unsqueeze(1).expand(-1, MM, -1)    # [B, MM, P]

        spike.scatter_add_(2, self.idx_floor.unsqueeze(0).expand(B, -1, -1), img * self.w_floor.unsqueeze(0))
        spike.scatter_add_(2, self.idx_ceil.unsqueeze(0).expand(B, -1, -1),  img * self.w_ceil.unsqueeze(0))

        # Grouped conv1d: one filter (the pulse) per (batch × channel) stream
        spike = spike.reshape(1, B * MM, self.buffer_len)
        pk    = self.base_pulse.view(1, 1, -1).expand(B * MM, 1, -1).contiguous()
        conv  = F.conv1d(spike, pk, padding=pk.shape[-1] // 2, groups=B * MM)

        # conv shape: [1, B*MM, T_out] -> [B, M_rx, M_tx, T]
        # index order: spike was [B, M*M, T] with M*M = rx*M + tx
        # so reshape to [B, M_rx, M_tx, T] and sum over M_tx (dim=2)
        conv = conv.view(B, self.M, self.M, -1)[:, :, :, :self.buffer_len]
        return conv.sum(dim=2)                                     # [B, M_rx, N_t]

    # ------------------------------------------------------------------
    # 2. DAS adjoint  (delay-and-align, two outputs)
    # ------------------------------------------------------------------
    def das_adjoint(self, rf_data):
        """rf_data: [B, M_rx, T]
        returns:
            das_image        [B, nx*nz]   -- standard DAS (uniform weights)
            pre_summed       [B, M*M, nx*nz] -- per channel-pair contributions
                                               BEFORE the final summation.
                                               This is the MLP input.

        pre_summed[b, rx*M+tx, pixel] is the amplitude-weighted,
        delay-compensated contribution of the (tx, rx) element pair to
        that pixel — exactly the "aligned channel values" the ABLE MLP
        uses to predict apodization weights (professor_meeting_context.md).
        """
        B  = rf_data.shape[0]
        MM = self.M * self.M

        # Pad rf_data into the full buffer and replicate each rx signal
        # across all M assumed tx positions (index order: rx*M + tx)
        v = rf_data.new_zeros(B, MM, self.buffer_len)
        v[:, :, :rf_data.shape[-1]] = (
            rf_data.unsqueeze(2)
                   .expand(-1, -1, self.M, -1)
                   .reshape(B, MM, -1)
        )

        val_f = torch.gather(v, 2, self.idx_floor.unsqueeze(0).expand(B, -1, -1))
        val_c = torch.gather(v, 2, self.idx_ceil.unsqueeze(0).expand(B, -1, -1))

        pre_summed = val_f * self.w_floor.unsqueeze(0) + val_c * self.w_ceil.unsqueeze(0)
        # pre_summed: [B, M*M, nx*nz]

        das_image = pre_summed.sum(dim=1)   # uniform weights -> DAS  [B, nx*nz]
        return das_image, pre_summed

    # ------------------------------------------------------------------
    # 3. True adjoint of simulate  (for iterative solvers like FISTA)
    # ------------------------------------------------------------------
    def simulate_adjoint(self, rf_data):
        """True mathematical adjoint of simulate(): matched filter + DAS.

        simulate() applies scatter_add then conv1d(pulse). The adjoint
        must reverse both: corr(rf, pulse) = conv1d(rf, flip(pulse)) first,
        then gather with the same delay/gain weights.

        das_adjoint() skips the pulse correlation (intentionally, because
        the MLP was trained on those unfiltered pre_summed values). Using
        das_adjoint as the FISTA gradient gives a 60% inner-product
        mismatch, causing guaranteed divergence. This method passes the
        adjoint test and is used only by FISTA.

        Returns: image [B, nx*nz]  (no pre_summed — FISTA does not need it)
        """
        B, M, T = rf_data.shape
        MM = M * M

        # Cross-correlate each receive channel with the pulse (adjoint of conv1d)
        pl = self.base_pulse.flip(0).view(1, 1, -1)
        pl_exp = pl.expand(B * M, 1, -1).contiguous()
        rf_mf = F.conv1d(
            rf_data.reshape(1, B * M, T),
            pl_exp,
            padding=pl.shape[-1] // 2,
            groups=B * M,
        ).view(B, M, -1)[..., :self.buffer_len]

        # Gather delay-compensated samples (same indices / weights as das_adjoint)
        v = rf_mf.new_zeros(B, MM, self.buffer_len)
        v[:, :, :rf_mf.shape[-1]] = (
            rf_mf.unsqueeze(2)
                 .expand(-1, -1, self.M, -1)
                 .reshape(B, MM, -1)
        )
        val_f = torch.gather(v, 2, self.idx_floor.unsqueeze(0).expand(B, -1, -1))
        val_c = torch.gather(v, 2, self.idx_ceil.unsqueeze(0).expand(B, -1, -1))
        pre   = val_f * self.w_floor.unsqueeze(0) + val_c * self.w_ceil.unsqueeze(0)
        return pre.sum(dim=1)   # [B, nx*nz]
