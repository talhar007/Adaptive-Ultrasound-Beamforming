"""Deepwave-based forward model — wave-equation alternative to ForwardModel.

Same API as ForwardModel (drop-in via --forward deepwave), but simulate()
solves the 2-D scalar wave equation with finite differences using Deepwave
(https://github.com/ar4/deepwave, Richardson 2023) in Born mode:

    background: uniform speed-of-sound c (same as the analytic model)
    scatterers: velocity perturbations on the imaging grid
    scalar_born propagates each transmit shot through the background and
    records the singly-scattered field at all receive elements.

Why Born mode: it is linear in the scatterer map, exactly like the analytic
model (per-scatterer spikes + 1/r gain), so the two models differ only in
PHYSICS FIDELITY — deepwave adds true diffraction and finite-aperture
effects from wave propagation — not in problem structure. That makes it a
clean physics-mismatch testbed: generate RF with deepwave, reconstruct with
the analytic operators (DAS / FISTA / MVDR / ABLE all inherited unchanged),
and compare against the analytic-data results.

Grid supersampling (IMPORTANT): the image pixel pitch (~0.50 mm) gives only
~2.4 FD nodes per wavelength at 5 MHz / 5920 m/s — far below the ~6/lambda
a 4th-order scheme needs, and numerical dispersion then slows the wave to
~c/2 (measured, not theoretical: arrivals came in at 1.93x the geometric
delay). The FD grid therefore runs at `supersample` x the image resolution
(default 3 -> ~7 nodes/lambda) with accuracy=8, which brings arrival-time
errors down to the sub-sample range. Image pixels land exactly on every
`supersample`-th node, so no interpolation of the scatterer map is needed.

Remaining approximations:
  - element positions snap to the nearest FD node (max error h/2 ~ 0.08 mm);
  - recorded traces are advanced by half a pulse length so echo peaks land
    at the analytic model's geo_delay samples (das_adjoint stays valid);
  - RF is peak-normalized per sample (Born amplitude scale is arbitrary;
    every consumer of rf here is scale-invariant or adaptive).
"""
import torch

from .forward_model import ForwardModel


class DeepwaveForwardModel(ForwardModel):

    def __init__(self, M=64, pitch=1e-3, c=5920.0, fc=5e6, fs=40e6,
                 nx=128, nz=128, x_lim=32e-3, z_start=10e-3, z_end=74e-3,
                 device='cpu', pml_width=20, scatter_scale=0.1,
                 supersample=3):
        super().__init__(M=M, pitch=pitch, c=c, fc=fc, fs=fs, nx=nx, nz=nz,
                         x_lim=x_lim, z_start=z_start, z_end=z_end,
                         device=device)
        import deepwave                      # deferred: optional dependency
        self._deepwave = deepwave
        self.pml_width = pml_width
        self.scatter_scale = scatter_scale   # velocity perturbation = scale*c*map
        self.ss = supersample

        # ---- FD grid: image pixel pitch / supersample ----
        h_x = 2 * x_lim / (nx - 1)
        h_z = (z_end - z_start) / (nz - 1)
        assert abs(h_x - h_z) < 1e-9, "image pixels must be square for the FD grid"
        self.h = h_x / self.ss
        pts_per_wavelength = (c / fc) / self.h
        assert pts_per_wavelength >= 6, (
            f"FD grid too coarse: {pts_per_wavelength:.1f} nodes/wavelength "
            f"(<6 causes severe numerical dispersion) — raise supersample")

        # Rows: [absorbing pad | z=0 (elements) ... z_start ... z_end]
        # The pad keeps the elements out of the top PML region.
        self.pad_top = pml_width + 4
        j0 = int(round(z_start / self.h))    # FD rows between elements and image
        self.row_img0 = self.pad_top + j0    # first image row (FD units)
        n_rows = self.row_img0 + (nz - 1) * self.ss + 1
        n_cols = (nx - 1) * self.ss + 1
        self.register_buffer('v_bg', torch.full((n_rows, n_cols), float(c),
                                                device=device))

        # ---- element locations, snapped to FD nodes ----
        elem_x = (torch.arange(M, device=device) - (M - 1) / 2) * pitch
        cols = torch.round((elem_x + x_lim) / self.h).long().clamp(0, n_cols - 1)
        row_e = self.pad_top
        src = torch.stack([torch.full_like(cols, row_e), cols], dim=-1)
        self.register_buffer('src_loc', src.unsqueeze(1))            # [M, 1, 2]
        rec = torch.stack([torch.full_like(cols, row_e), cols], dim=-1)
        self.register_buffer('rec_loc', rec.unsqueeze(0).expand(M, -1, -1)
                                           .contiguous())            # [M, M, 2]

        # ---- source wavelet: same base pulse as the analytic model ----
        Lp = self.base_pulse.shape[0]
        self.pulse_half = Lp // 2
        nt = self.buffer_len + self.pulse_half + 1
        amp = torch.zeros(M, 1, nt, device=device)
        amp[:, 0, :Lp] = self.base_pulse
        self.register_buffer('src_amp', amp)

    # ------------------------------------------------------------------
    # simulate() override: FD wave propagation, one Born solve per sample
    # ------------------------------------------------------------------
    def _born(self, scatterer_maps, src_amp):
        """Shared Born-propagation loop for a given set of per-shot source
        amplitudes. scatterer_maps: [B, nx*nz] -> rf [B, M, buffer_len]."""
        B = scatterer_maps.shape[0]
        dt = 1.0 / self.fs
        rf_out = []
        for b in range(B):
            scatter = torch.zeros_like(self.v_bg)
            scatter[self.row_img0::self.ss, ::self.ss] = (
                self.scatter_scale * self.c
                * scatterer_maps[b].view(self.nz, self.nx)
            )
            out = self._deepwave.scalar_born(
                self.v_bg, scatter, self.h, dt,
                source_amplitudes=src_amp,
                source_locations=self.src_loc,
                receiver_locations=self.rec_loc,
                pml_width=self.pml_width,
                accuracy=8,
            )
            rec = out[-1]                                    # [M shots, M rx, nt]
            rf = rec.sum(dim=0)                              # compound tx -> [M, nt]
            # Advance by half a pulse so echo peaks sit at geo_delay samples
            rf = rf[:, self.pulse_half:self.pulse_half + self.buffer_len]
            rf = rf / (rf.abs().max() + 1e-12)
            rf_out.append(rf)
        return torch.stack(rf_out, dim=0)                    # [B, M, buffer_len]

    def simulate(self, scatterer_maps):
        """scatterer_maps: [B, nx*nz] -> rf_data [B, M, buffer_len]

        Pixel ordering is the physics-grid order ([nz, nx] flattened);
        image pixel (j, k) sits on FD node (row_img0 + j*ss, k*ss).
        """
        return self._born(scatterer_maps, self.src_amp)

    def simulate_tx(self, scatterer_maps, w_tx=None, tau_tx=None):
        """Transmit-side TX parameters in the FD simulator (TEST-time use):
        shot m fires with its wavelet scaled by w_tx[m] and time-shifted by
        tau_tx[m] samples (fractional shift via linear interpolation of the
        source amplitude series). This is the wave-equation counterpart of
        ForwardModel.simulate_tx — the learned firing strategy is applied
        to the SOURCES, before propagation, exactly as hardware would.
        """
        amp = self.src_amp
        M, _, nt = amp.shape
        if tau_tx is not None:
            # y_m(t) = x(t - tau_m): sample the original wavelet at t - tau
            idx = torch.arange(nt, device=amp.device, dtype=amp.dtype)
            pos = idx.view(1, 1, -1) - tau_tx.view(M, 1, 1)
            p0  = pos.floor().long().clamp(0, nt - 1)
            f   = (pos - pos.floor()).clamp(0, 1)
            p1  = (p0 + 1).clamp(0, nt - 1)
            base = amp.expand(M, 1, nt)
            valid = ((pos >= 0) & (pos <= nt - 1)).to(amp.dtype)
            amp = (base.gather(2, p0) * (1 - f) + base.gather(2, p1) * f) * valid
        if w_tx is not None:
            amp = amp * w_tx.view(M, 1, 1)
        return self._born(scatterer_maps, amp)
