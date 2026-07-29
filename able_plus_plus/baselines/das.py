"""Standard Delay-and-Sum baseline (RQ3).

Re-uses the forward model's ``das_adjoint`` operator (dynamic TX+RX
focusing), then sums the channel-pair contributions with a HANNING
apodization taper by default — matching the paper's DAS baseline
("Delay-and-sum (DAS) beamforming with Hanning apodization", Luijten et
al. Fig. 6). A uniform boxcar sum leaks strong sidelobes around every
bright scatterer (visible as structured clutter on sparse scenes); the
Hann taper trades a slightly wider mainlobe for far lower sidelobes.

Pass apodization=None for the plain boxcar (uniform-weight) variant.
"""
import torch


@torch.no_grad()
def das_reconstruct(model, rf_data, apodization='hann'):
    """model: ForwardModel    rf_data: [B, M, T]
    apodization: 'hann' (default, paper's DAS baseline) | None (boxcar)
    returns: flattened reconstructed image [B, nx*nz]
    """
    das_image, pre_summed = model.das_adjoint(rf_data)
    if apodization is None:
        return das_image

    M = model.M
    win = torch.hann_window(M, periodic=False,
                            device=rf_data.device, dtype=rf_data.dtype)
    # separable 2-D taper over the (rx, tx) channel-pair grid,
    # normalized so the total weight equals the boxcar's (unity gain)
    w2d = (win.view(M, 1) * win.view(1, M)).reshape(M * M)
    w2d = w2d * (M * M / w2d.sum())
    return (pre_summed * w2d.view(1, -1, 1)).sum(dim=1)
