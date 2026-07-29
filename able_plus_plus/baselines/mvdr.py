"""MVDR / Eigen-Based Minimum Variance baseline (Luijten et al. Sec. II-C).

Per pixel, the Minimum Variance Distortionless Response beamformer picks
apodization weights that minimize output power subject to unity gain toward
the (already time-of-flight corrected) signal, Eq. (6):

    min_w  w^H R w   s.t.  w^H 1 = 1     ->     w = R^-1 1 / (1^T R^-1 1)

This is the classical adaptive beamformer that ABLE was designed to
approximate with a neural network, which makes it the most important
non-learned comparison point.

Implementation follows the paper:
  - The per-pixel receive-aperture vector y[x,z] in R^M is obtained from the
    ForwardModel's das_adjoint pre_summed tensor ([B, M*M, P], laid out
    rx-major: index = rx*M + tx) by summing over the transmit dimension.
  - The covariance matrix R is estimated with spatial smoothing over
    subapertures of length L (Eqs. 8-9) — averaging outer products of all
    N-L+1 sliding subapertures — which both decorrelates coherent sources
    and reduces the inversion from M x M to L x L.
  - Diagonal loading eps = D * trace(R) stabilizes the inversion (Eq. 10).
  - eigen=True applies the Eigen-Based MV refinement (Eq. 11): the weights
    are projected onto the dominant eigen-subspace of R (top eig_frac of
    eigenvectors, paper uses ~50%), improving contrast at extra cost.
"""
import torch


@torch.no_grad()
def mvdr_reconstruct(model, rf_data, L=None, diag_load=0.01,
                     eigen=False, eig_frac=0.2, smoothing=True,
                     loading='eigen', kappa=1e3):
    """MVDR reconstruction of every pixel, batched over the whole image.

    rf_data:   [B, M, T] simulated channel data
    L:         subaperture length for spatial smoothing (default M//2,
               the common choice; paper Table II uses L=32 for N=128)
    smoothing: False = variant WITHOUT spatial smoothing: the full
               M-element aperture with a single snapshot, so R is the
               rank-1 outer product y y^T and all inversion stability
               comes from the diagonal loading. Maximum aperture (best
               potential resolution), minimum robustness.
    loading:   'eigen' (default) — eigenvalue-adaptive: per pixel choose
               the smallest eps that caps the condition number of
               R + eps*I at kappa:
                   eps = max(0, (lmax - kappa*lmin) / (kappa - 1)).
               Well-conditioned pixels get (almost) no loading — full
               adaptivity; ill-conditioned ones get exactly enough.
               'trace' — the paper's fixed rule eps = D * trace(R)
               (Eq. 10): scales with total energy (trace = sum of
               eigenvalues) but ignores the eigenvalue SPREAD.
    diag_load: D for loading='trace' (paper Table II: 0.1)
    kappa:     condition-number cap for loading='eigen'
    eigen:     apply the EBMV eigen-subspace projection (Eq. 11)

    returns: flat reconstructed image [B, nx*nz]
    """
    _, pre_summed = model.das_adjoint(rf_data)              # [B, M*M, P]
    B, MM, P = pre_summed.shape
    M = int(MM ** 0.5)
    L = M if not smoothing else (L or M // 2)
    n_sub = M - L + 1

    # Receive-aperture response per pixel: sum tx contributions out of the
    # rx-major [rx, tx] channel-pair axis (compounded synthetic aperture).
    y = pre_summed.reshape(B, M, M, P).sum(dim=2)           # [B, M, P]
    y = y.permute(0, 2, 1).contiguous()                     # [B, P, M]

    # Spatially smoothed covariance (Eqs. 8-9): sliding subapertures.
    sub = y.unfold(dimension=-1, size=L, step=1)            # [B, P, n_sub, L]
    R = torch.einsum('bpsl,bpsk->bplk', sub, sub) / n_sub   # [B, P, L, L]

    # Diagonal loading — two schemes, both with an absolute floor tied to
    # the image-wide signal level so signal-free pixels (R ~ 0) still yield
    # an invertible matrix: there the loading dominates, R ~ eps*I, and w
    # collapses to uniform 1/L — plain DAS on (near-)zero data, the correct
    # limit.
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)               # [B, P]
    if loading == 'eigen':
        # Eigenvalue-adaptive: smallest eps such that the loaded matrix has
        # condition number <= kappa. lmin is clamped at 0 (symmetric PSD up
        # to roundoff; tiny negative eigenvalues would give negative eps).
        evals = torch.linalg.eigvalsh(R)                    # [B, P, L] ascending
        lmin  = evals[..., 0].clamp_min(0.0)
        lmax  = evals[..., -1]
        load  = ((lmax - kappa * lmin) / (kappa - 1.0)).clamp_min(0.0)
    else:                                                   # 'trace' (Eq. 10)
        load = diag_load * tr
    load = load + 1e-8 * tr.mean(dim=-1, keepdim=True) + 1e-20
    eye  = torch.eye(L, device=R.device, dtype=R.dtype)
    R    = R + load[..., None, None] * eye

    # w = R^-1 a / (a^H R^-1 a) with steering vector a = 1 (Eq. 7).
    ones  = torch.ones(B, P, L, 1, device=R.device, dtype=R.dtype)
    Rinv1 = torch.linalg.solve(R, ones)[..., 0]             # [B, P, L]
    w     = Rinv1 / (Rinv1.sum(dim=-1, keepdim=True) + 1e-12)

    if eigen:
        # EBMV (Eq. 11): project w onto the dominant eigen-subspace of R.
        _, evecs = torch.linalg.eigh(R)                     # ascending order
        k  = max(1, int(eig_frac * L))
        Es = evecs[..., -k:]                                # [B, P, L, k]
        w  = torch.einsum('bplk,bpmk,bpm->bpl', Es, Es, w)

    # Beamform: average the weighted subaperture outputs (Eq. 12 with
    # spatial smoothing), preserving the distortionless unity-gain response.
    return torch.einsum('bpl,bpsl->bps', w, sub).mean(dim=-1)   # [B, P]
