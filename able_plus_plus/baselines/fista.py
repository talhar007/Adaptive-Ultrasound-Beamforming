"""FISTA baseline (RQ3): sparse-regularised iterative reconstruction

    min_x  (1/2) || A x - b ||_2^2  +  lambda * || x ||_1

solved with the Fast Iterative Shrinkage-Thresholding Algorithm, using the
project's *own* differentiable forward / adjoint pair as the linear operator
A / A^T. This is the "strong non-learned baseline representative of the
compressive sensing literature" called for in the proposal, and reusing
``simulate`` / ``das_adjoint`` here is a direct payoff of having built a
faithful differentiable forward model (RQ1) -- the same operator pair drives
the data generator, the DAS baseline, FISTA, and the learned pipeline.

Fairness note: FISTA uses the same ForwardModel as training but with no
learned parameters — it is a purely iterative, non-learned solver, making
it a valid comparison point for RQ3.
"""
import torch


def _soft_threshold(x, thresh):
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)


@torch.no_grad()
def fista_reconstruct(model, rf_data, n_iter=50, lam=1e-3, step=1e-2):
    """model: ForwardModel (provides A = simulate, A^T = das_adjoint[0])
    rf_data: [B, M, T]
    returns: flat reconstructed image [B, nx*nz]
    """
    B = rf_data.shape[0]
    P = model.nx * model.nz

    x = torch.zeros(B, P, device=rf_data.device, dtype=rf_data.dtype)
    z = x.clone()
    t = 1.0

    for _ in range(n_iter):
        residual = model.simulate(z) - rf_data
        grad, _ = model.das_adjoint(residual)               # A^T (A z - b)  [B, P]
        x_new = _soft_threshold(z - step * grad, lam * step)

        t_new = (1.0 + (1.0 + 4.0 * t ** 2) ** 0.5) / 2.0
        z     = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t  = x_new, t_new

    return x
