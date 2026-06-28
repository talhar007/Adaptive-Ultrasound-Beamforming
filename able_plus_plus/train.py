"""Training loop for the ABLE++ receive beamforming network.

Pipeline per step (professor meeting, 2026-06-xx):

    1. Draw a random scatterer map (ground truth)               [B, nx*nz]
    2. Forward model (FIXED, no_grad) -> RF channel data        [B, M, T]
    3. DAS adjoint -> pre_summed_samples                        [B, M*M, P]
    4. Reshape to pixel-wise MLP input                          [B*P, M*M]
    5. MLP predicts apodization weights                         [B*P, M*M]
    6. Weighted sum -> reconstructed image                      [B, P]
    7. Loss = SMSLE + unity-gain penalty; backprop into MLP only

No joint TX training stage — the forward model is a frozen physics engine.
"""
import torch

from .data.simulate import make_batch
from .networks.able_mlp import apply_mlp
from .networks.losses import total_loss


def train_step(model, mlp, optimizer, batch_size, lam=0.8, noise_level=0.05, device='cpu'):
    """Single optimisation step.

    Returns a dict with scalar loss values (for logging / history).
    """
    optimizer.zero_grad()

    gt_images, rf_data = make_batch(model, batch_size, noise_level=noise_level, device=device)

    # Forward model is fixed — das_adjoint runs under autograd so that
    # gradients CAN propagate through the interpolation back to pre_summed,
    # but since pre_summed only depends on (fixed) buffers and the (no_grad)
    # rf_data, all gradient flow stays inside the MLP.
    _, pre_summed = model.das_adjoint(rf_data)           # [B, M*M, P]

    p_recon, weights, _ = apply_mlp(mlp, pre_summed)    # [B, P], [B*P, M*M]

    target = gt_images.detach()                          # [B, P]
    loss, l_img, l_unity = total_loss(p_recon, target, weights, lam=lam)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'loss':         loss.item(),
        'image_loss':   l_img.item(),
        'unity_loss':   l_unity.item(),
    }


def train(model, mlp, n_steps, batch_size, lam=0.8, lr=1e-3,
          noise_level=0.05, device='cpu', callback=None):
    """Train the MLP for n_steps steps.

    callback(step, metrics) is called after every step if provided —
    use it for logging or checkpointing.

    Returns list of per-step metric dicts.
    """
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    history = []
    for i in range(1, n_steps + 1):
        metrics = train_step(model, mlp, optimizer, batch_size,
                             lam=lam, noise_level=noise_level, device=device)
        metrics['step'] = i
        history.append(metrics)
        if callback is not None:
            callback(i, metrics)
    return history
