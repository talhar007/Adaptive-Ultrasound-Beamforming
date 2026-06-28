"""Standard Delay-and-Sum baseline (RQ3).

Re-uses the forward model's ``das_adjoint`` operator, which performs dynamic
TX+RX focusing with uniform (boxcar) apodization weights -- exactly the
conventional, non-adaptive beamformer that ABLE / ABLE++ are compared
against in the proposal and in Luijten et al.
"""
import torch


@torch.no_grad()
def das_reconstruct(model, rf_data):
    """model: ForwardModel    rf_data: [B, M, T]
    returns: flattened reconstructed image [B, nx*nz]
    """
    das_image, _ = model.das_adjoint(rf_data)
    return das_image
