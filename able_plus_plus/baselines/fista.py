"""FISTA baseline (RQ3): sparse-regularised iterative reconstruction
    A much stronger non-learned baseline from the compressed-sensing world. It solves the optimization problem
    min_x  (1/2) || A x - b ||_2^2  +  lambda * || x ||_1

Fairness note: FISTA uses the same ForwardModel as training but with no
learned parameters — it is a purely iterative, non-learned solver, making
it a valid comparison point for RQ3.
"""
import torch

# shrinks every pixel toward zero by λ·step, which is what produces those very clean, sparse FISTA images.
def _soft_threshold(x, thresh):
    return torch.sign(x) * torch.relu(torch.abs(x) - thresh)

# power iteration on AᵀA to estimate the operator norm, which sets the gradient step size 1/L so convergence is guaranteed regardless of data scale.
@torch.no_grad()
def _estimate_lipschitz(model, device, dtype, n_power_iter=30):
    P = model.nx * model.nz
    v = torch.randn(1, P, device=device, dtype=dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(n_power_iter):
        Av   = model.simulate(v)
        AtAv = model.simulate_adjoint(Av)
        v    = AtAv / (AtAv.norm() + 1e-12)
    Av   = model.simulate(v)
    AtAv = model.simulate_adjoint(Av)
    L = (v * AtAv).sum().abs().item()
    return max(L, 1e-12)


@torch.no_grad()
def fista_reconstruct(model, rf_data, n_iter=10):
    """FISTA using model.simulate_adjoint as the gradient operator.

    simulate_adjoint applies matched filtering (corr with base_pulse) before
    the delay-and-sum, making it the true mathematical adjoint of simulate.

    Lambda is adaptive: set to 0.05 * peak(A^T b), computed from the actual
    RF measurement, and held fixed across all iterations (no annealing, so
    FISTA retains meaningful L1 regularisation throughout and stays a
    realistic compressed-sensing baseline).

    n_iter defaults to 10 (was 50): comparable memory/compute budget to the
    other methods, at the cost of less complete convergence.

    rf_data: [B, M, T]
    returns: flat reconstructed image [B, nx*nz]
    """
    B = rf_data.shape[0]
    P = model.nx * model.nz
    device, dtype = rf_data.device, rf_data.dtype

    step = 1.0 / _estimate_lipschitz(model, device, dtype)

    x = torch.zeros(B, P, device=device, dtype=dtype)
    z = x.clone()
    t = 1.0

    # Data-driven lambda: 0.05 × peak of the matched-filtered back-projection.
    # Low enough to recover scatterers down to ~5% of the brightest response
    # (earlier 0.5/0.3 × peak thresholds cut weak-but-real scatterers out of
    # clustered fields), while still suppressing background clutter.
    atb = model.simulate_adjoint(rf_data)
    lam = 0.05 * atb.abs().amax(dim=-1).mean().item()

    for _ in range(n_iter):
        residual = model.simulate(z) - rf_data
        grad     = model.simulate_adjoint(residual)
        x_new    = _soft_threshold(z - step * grad, lam * step)

        t_new = (1.0 + (1.0 + 4.0 * t ** 2) ** 0.5) / 2.0
        z     = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t  = x_new, t_new

    return x
