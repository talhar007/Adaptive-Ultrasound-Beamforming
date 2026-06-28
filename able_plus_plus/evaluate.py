"""Quantitative evaluation metrics -- FWHM, CNR, MAE -- matching the
"Evaluation Metrics" section of the proposal and Sec. IV-B of Luijten et al.,
so that DAS / FISTA / ABLE / ABLE++ can be compared on a common footing (RQ3).
"""
import numpy as np
import torch


def mae(pred, target):
    """Mean Absolute Error between a reconstruction and its reference
    (e.g. an EBMV target), as used for "reconstruction fidelity"."""
    return (pred - target).abs().mean().item()


def fwhm_1d(profile, dx):
    """Full-Width-at-Half-Maximum of a 1-D beam profile (linear amplitude).

    profile: 1-D array centred roughly on its peak; dx: sample spacing.
    """
    profile = np.asarray(profile)
    half = profile.max() / 2.0
    above = np.where(profile >= half)[0]
    if len(above) < 2:
        return float('nan')
    return (above[-1] - above[0] + 1) * dx


def fwhm_point_scatterer(image_2d, peak_rc, dx, dz):
    """Lateral & axial FWHM through a simulated point scatterer (Sec. IV-B).

    image_2d: [nz, nx] linear-amplitude reconstruction
    peak_rc:  (row, col) index of the scatterer peak
    """
    r, c = peak_rc
    return fwhm_1d(image_2d[r, :], dx), fwhm_1d(image_2d[:, c], dz)


def cnr(image_db, mask_low, mask_high):
    """Contrast-to-Noise Ratio (Eq. 18):

        CNR = 20 log10( |mu_low - mu_high| / sqrt((var_low + var_high)/2) )

    image_db: log-compressed (dB) reconstruction; mask_low/mask_high: boolean
    region masks of the same shape, e.g. drawn on simulated anechoic cysts.
    """
    low, high = image_db[mask_low], image_db[mask_high]
    num = abs(low.mean() - high.mean())
    den = np.sqrt((low.var() + high.var()) / 2.0)
    return float(20 * np.log10(num / den + 1e-12))


@torch.no_grad()
def envelope_db(image_2d):
    """Hilbert-envelope, log-compressed (dB, normalised to peak) version of
    a beamformed image -- the standard B-mode display transform used
    throughout Luijten et al. and the supplied notebook."""
    from scipy.signal import hilbert
    img = image_2d.detach().cpu().numpy() if torch.is_tensor(image_2d) else np.asarray(image_2d)
    env = np.abs(hilbert(img, axis=0))
    return 20 * np.log10(env / env.max() + 1e-12)
