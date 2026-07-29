"""Loss terms for ABLE++ training.

Composite objective:

    L_total = lambda * L_SMSLE + (1 - lambda) * L_constraints      (Eq. 15)

L_SMSLE: signed mean-squared logarithmic error between the MLP-beamformed
image and the ground-truth scatterer map (handles large dynamic range).

L_constraints: unity-gain penalty on the receive apodization weights
(the only constraint after the professor meeting — TX penalties removed
because TX parameters are no longer learned).
"""
import torch


def smsle_loss(p_pred, p_target, eps=1e-6):
    """Signed-Mean-Squared-Logarithmic-Error on peak-normalized images,
    class-balanced between scatterer and background pixels.

    Both images are normalized per sample by their own peak |amplitude|
    before the log-domain comparison — the SAME normalization the
    evaluation / demo pipeline applies to GT, DAS, FISTA and ABLE maps, so
    the training objective and the reported MAE live on one common scale.

    Class balancing: a sparse field has ~3-20 scatterer pixels against
    ~16k background pixels. A plain mean over pixels is >99.9% background,
    so the optimizer learns that crushing weak scatterers to zero is a
    profitable trade for slightly cleaner background — which showed up as
    "missing" points in the reconstructions. Averaging the scatterer and
    background pools separately (50/50) makes deleting a true scatterer
    exactly as expensive as hallucinating a false one.

    p_pred, p_target: [B, nx*nz]
    """
    pred_scale = p_pred.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    targ_scale = p_target.abs().amax(dim=-1, keepdim=True).clamp(min=eps)

    p_pred_norm = p_pred / pred_scale
    p_target_norm = p_target / targ_scale

    pred_pos, pred_neg = torch.relu(p_pred_norm), torch.relu(-p_pred_norm)
    targ_pos, targ_neg = torch.relu(p_target_norm), torch.relu(-p_target_norm)

    sq = 0.5 * ((torch.log10(pred_pos + eps) - torch.log10(targ_pos + eps)).pow(2)
                + (torch.log10(pred_neg + eps) - torch.log10(targ_neg + eps)).pow(2))

    fg = p_target.abs() > 0
    n_fg = fg.sum()
    if n_fg == 0 or n_fg == fg.numel():
        return sq.mean()
    loss_fg = sq[fg].mean()
    loss_bg = sq[~fg].mean()
    return 0.5 * (loss_fg + loss_bg)


def unity_gain_penalty(weights):
    """Apodization weights should sum to ~1 per pixel (distortionless response).

    With the linear MLP output this is the only thing anchoring the weight
    scale, so it must NOT be shrunk by 1/N — dividing by N=M*M=4096 inside
    the square scaled the penalty by ~6e-8 and made it a no-op.

    weights: [B*P, M*M]  — sum over the channel-pair dimension (last dim)
    """
    weight_sums = weights.sum(dim=-1)  # [B*P]
    return (weight_sums - 1.0).pow(2).mean()


def background_power(p_pred, p_target, eps=1e-6):
    """Linear-domain background energy: mean squared (peak-normalized)
    amplitude on pixels where the ground truth is empty.

    SMSLE compares in the LOG domain, so residual clutter at, say, 1% of
    peak is nearly free there — yet it is exactly what the reported CNR
    and PSNR punish. This term aligns training with those metrics: it
    charges linear-scale energy for anything reconstructed where nothing
    exists. Weighted by lam_bg in total_loss (0 = off, keeps the paper's
    objective untouched).
    """
    scale = p_pred.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    norm = p_pred / scale
    bg = p_target.abs() <= 0
    if bg.sum() == 0:
        return p_pred.new_zeros(())
    return norm[bg].pow(2).mean()


def tx_smoothness_loss(w_tx, tau_tx, max_delay):
    """Total-variation penalty across element index on the learned transmit
    apodization and firing delays — discourages the jagged, element-to-
    element sign-flipping solution the optimizer finds when nothing
    constrains it (confirmed on checkpoints_pp5: roughness = mean(diff^2)/
    var of the learned w_tx/tau_tx was 2.32/2.20, ROUGHER than i.i.d. random
    noise (1.72) and ~200x rougher than a smooth Hann taper (0.01)).

    Rationale (classical array/beamforming theory): aperture-weighting
    smoothness directly controls transmit sidelobe level — it's why real
    apodization windows (Hann, Hamming, Tukey) are always smooth functions
    of element position. An unconstrained per-element w_tx/tau_tx has no
    reason to stay smooth; the resulting excess sidelobe energy is
    invisible on isolated scatterers (lands on background, which lam_bg
    suppresses hard) but shows up as visible clutter between CLOSE
    scatterers, where sidelobe leakage from one point lands on a real
    neighboring echo and can't be suppressed without risking the signal.

    tau_tx's term is normalized by max_delay^2 so it sits on the same O(1)
    scale as w_tx's term (already bounded via tanh to (-1,1)), keeping one
    lam_tx_smooth weight meaningful regardless of the configured bound.
    """
    w_term = (w_tx[1:] - w_tx[:-1]).pow(2).mean()
    tau_term = (tau_tx[1:] - tau_tx[:-1]).pow(2).mean() / (max_delay ** 2)
    return w_term + tau_term


def total_loss(p_pred, p_target, weights, lam=0.8, lam_bg=0.0):
    """L_total = lam * L_SMSLE + (1 - lam) * L_unity + lam_bg * L_bg

    (Eq. 15 of the paper, optionally extended with the linear-domain
    background-suppression term that aligns training with CNR/PSNR.)

    p_pred   : [B, P]      MLP-beamformed reconstruction
    p_target : [B, P]      ground-truth scatterer map (detached)
    weights  : [B*P, M*M]  apodization weights from MLP (for unity penalty)
    lam      : float       weight on the image loss term
    lam_bg   : float       weight on the background-power term (default 0)

    returns (total, image_loss, unity_loss) for logging.
    """
    l_img   = smsle_loss(p_pred, p_target)
    l_unity = unity_gain_penalty(weights)
    total = lam * l_img + (1.0 - lam) * l_unity
    if lam_bg > 0:
        total = total + lam_bg * background_power(p_pred, p_target)
    return total, l_img, l_unity
