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
    """Signed-Mean-Squared-Logarithmic-Error.

    Splits data into positive/negative halves and compares in the log domain,
    matching how ultrasound B-mode images are compressed for display without
    discarding the bipolar RF sign information.

    p_pred, p_target: any broadcastable shape (typically [B, nx*nz])
    """
    pred_pos, pred_neg = torch.relu(p_pred),  torch.relu(-p_pred)
    targ_pos, targ_neg = torch.relu(p_target), torch.relu(-p_target)

    loss_pos = (torch.log10(pred_pos + eps) - torch.log10(targ_pos + eps)).pow(2).mean()
    loss_neg = (torch.log10(pred_neg + eps) - torch.log10(targ_neg + eps)).pow(2).mean()
    return 0.5 * (loss_pos + loss_neg)


def unity_gain_penalty(weights):
    """Apodization weights should sum to ~1 per pixel (distortionless response).

    weights: [B*P, M*M]  — sum over the channel-pair dimension (last dim)
    """
    return (weights.sum(dim=-1) - 1.0).pow(2).mean()


def total_loss(p_pred, p_target, weights, lam=0.8):
    """L_total = lambda * L_SMSLE + (1 - lambda) * L_unity  (Eq. 15)

    p_pred   : [B, P]      MLP-beamformed reconstruction
    p_target : [B, P]      ground-truth scatterer map (detached)
    weights  : [B*P, M*M]  apodization weights from MLP (for unity penalty)
    lam      : float       weight on the image loss term

    returns (total, image_loss, unity_loss) for logging.
    """
    l_img   = smsle_loss(p_pred, p_target)
    l_unity = unity_gain_penalty(weights)
    return lam * l_img + (1.0 - lam) * l_unity, l_img, l_unity
