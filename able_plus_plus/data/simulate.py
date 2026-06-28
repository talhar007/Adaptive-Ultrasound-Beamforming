"""Synthetic training / evaluation data generation.

The forward model acts as the data generator: random scatterer maps are
pushed through the fixed physics model to produce RF channel data, giving
exact ground truth for every training example with no real acquisitions.

Four scatterer types are supported so the MLP generalises to diverse
imaging scenarios — not just isolated point scatterers:

    'sparse'    (default) — a few isolated point reflectors (PSF testing)
    'dense'     — many overlapping scatterers (speckle / tissue-like)
    'clustered' — tight groups of points (lesion / cyst boundary testing)
    'mixed'     — randomly draw one of the three types per sample in a batch
"""
import random as _random

import torch


# --------------------------------------------------------------------------
# Per-image scatterer generators
# --------------------------------------------------------------------------

def _sparse(nx, nz, n_points=(2, 6), amplitude=(0.5, 1.0), device='cpu'):
    """2-6 isolated point reflectors."""
    img = torch.zeros(nx * nz, device=device)
    k   = torch.randint(n_points[0], n_points[1] + 1, (1,)).item()
    idx = torch.randint(0, nx * nz, (k,), device=device)
    amp = torch.empty(k, device=device).uniform_(*amplitude)
    img.scatter_add_(0, idx, amp)
    return img


def _dense(nx, nz, fill_frac=(0.1, 0.3), amplitude=(0.1, 1.0), device='cpu'):
    """Tissue-like speckle: many scatterers at random amplitudes."""
    img  = torch.zeros(nx * nz, device=device)
    frac = torch.empty(1).uniform_(*fill_frac).item()
    k    = max(1, int(frac * nx * nz))
    idx  = torch.randint(0, nx * nz, (k,), device=device)
    amp  = torch.empty(k, device=device).uniform_(*amplitude)
    img.scatter_add_(0, idx, amp)
    return img


def _clustered(nx, nz, n_clusters=(1, 4), cluster_r=(3, 8),
               n_per_cluster=(3, 10), amplitude=(0.5, 1.0), device='cpu'):
    """Tight groups of points — mimics echogenic inclusions."""
    img  = torch.zeros(nx * nz, device=device)
    nc   = torch.randint(n_clusters[0], n_clusters[1] + 1, (1,)).item()
    for _ in range(nc):
        cx = torch.randint(0, nx, (1,)).item()
        cz = torch.randint(0, nz, (1,)).item()
        r  = torch.randint(cluster_r[0], cluster_r[1] + 1, (1,)).item()
        k  = torch.randint(n_per_cluster[0], n_per_cluster[1] + 1, (1,)).item()
        for _ in range(k):
            dx = torch.randint(-r, r + 1, (1,)).item()
            dz = torch.randint(-r, r + 1, (1,)).item()
            x  = min(max(cx + dx, 0), nx - 1)
            z  = min(max(cz + dz, 0), nz - 1)
            amp = torch.empty(1, device=device).uniform_(*amplitude).item()
            img[x * nz + z] = img[x * nz + z] + amp
    return img


_GENERATORS = {
    'sparse':    _sparse,
    'dense':     _dense,
    'clustered': _clustered,
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def random_point_scatterers(batch_size, nx, nz, n_points=(2, 6),
                            amplitude=(0.5, 1.0), device='cpu'):
    """Sparse point scatterers — direct replacement for the old API.

    Returns [B, nx*nz] float tensor of ground-truth reflectivity.
    """
    images = torch.zeros(batch_size, nx * nz, device=device)
    for b in range(batch_size):
        images[b] = _sparse(nx, nz, n_points=n_points, amplitude=amplitude, device=device)
    return images


def random_scatterer_batch(batch_size, nx, nz, scatterer_type='mixed', device='cpu'):
    """Generate a batch of diverse scatterer maps.

    scatterer_type: 'sparse' | 'dense' | 'clustered' | 'mixed'
        'mixed' draws a fresh random type for every sample independently,
        which forces the MLP to generalise across imaging scenarios.

    Returns [B, nx*nz] float tensor of ground-truth reflectivity.
    """
    images = torch.zeros(batch_size, nx * nz, device=device)
    types  = list(_GENERATORS.keys())
    for b in range(batch_size):
        t   = _random.choice(types) if scatterer_type == 'mixed' else scatterer_type
        gen = _GENERATORS[t]
        images[b] = gen(nx, nz, device=device)
    return images


def make_batch(model, batch_size, noise_level=0.05, scatterer_type='mixed', device='cpu'):
    """Draw ground-truth scatterer maps, simulate RF data, add noise.

    The forward model runs under torch.no_grad() — data generation is a
    fixed physics step; only the subsequent MLP reconstruction accumulates
    gradients.

    Returns:
        gt_images [B, nx*nz]   ground-truth reflectivity
        rf_data   [B, M, T]    simulated channel data with noise
    """
    gt_images = random_scatterer_batch(batch_size, model.nx, model.nz,
                                       scatterer_type=scatterer_type, device=device)
    with torch.no_grad():
        rf_clean = model.simulate(gt_images)
        noise    = torch.randn_like(rf_clean) * noise_level * rf_clean.abs().max()
        rf_data  = rf_clean + noise
    return gt_images, rf_data
