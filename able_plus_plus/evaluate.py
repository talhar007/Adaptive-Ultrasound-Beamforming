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


def psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio (dB) between two maps on a shared
    [0, max_val] scale. Higher = better."""
    pred, target = np.asarray(pred), np.asarray(target)
    mse = float(((pred - target) ** 2).mean())
    return float(10.0 * np.log10(max_val ** 2 / (mse + 1e-12)))


def smsle(pred_norm, target_norm, floor=1e-3):
    """Signed-MSLE metric (Luijten et al. Eq. 16) on peak-normalized
    magnitude maps. Lower = better.

    Both inputs are non-negative on the shared [0, 1] scale, so the negative
    branch of Eq. 16 vanishes and the metric reduces to the MSLE of the
    positive part. Values are floored at `floor` (1e-3 = -60 dB, the B-mode
    display dynamic range) so empty background pixels — identical in pred
    and target — contribute zero rather than unbounded log(eps) noise.
    """
    p = np.clip(np.asarray(pred_norm), floor, None)
    t = np.clip(np.asarray(target_norm), floor, None)
    return float(((np.log10(p) - np.log10(t)) ** 2).mean())


def scatterer_region_masks(gt_norm, thresh=0.05, r_high=2, r_low=6):
    """High/low-intensity region masks for CNR, derived from ground truth.

    high = within r_high pixels of any true scatterer; low = background
    further than r_low pixels from every scatterer. Luijten et al. draw
    these regions by hand on cyst phantoms (Fig. 3); with synthetic data the
    exact ground truth lets us construct them automatically.
    """
    core = torch.from_numpy((np.asarray(gt_norm) > thresh).astype(np.float32))[None, None]

    def dilate(mask, r):
        return torch.nn.functional.max_pool2d(mask, kernel_size=2 * r + 1,
                                              stride=1, padding=r)

    high = dilate(core, r_high)[0, 0].numpy() > 0.5
    low  = dilate(core, r_low)[0, 0].numpy() < 0.5
    return low, high


def scatterer_cnr(image_db, gt_norm, **mask_kwargs):
    """CNR (Eq. 18) on a dB image, with region masks auto-derived from the
    ground-truth scatterer map. Higher = better contrast."""
    mask_low, mask_high = scatterer_region_masks(gt_norm, **mask_kwargs)
    if mask_low.sum() == 0 or mask_high.sum() == 0:
        return float('nan')
    return cnr(np.asarray(image_db), mask_low, mask_high)


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
