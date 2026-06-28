"""Neural receive apodization network — "Neural Apodization (ABLE MLP)" block.

After the professor meeting the MLP input is the pre_summed_samples tensor
from das_adjoint, NOT the M-dimensional receive-aligned signal.  Each pixel
has M*M = 4096 aligned channel-pair values; the MLP maps these to 4096
apodization weights, and a weighted sum replaces the uniform DAS sum.

Input/output dimension N = M*M (= 4096 for the standard M=64 array).

Architecture (Luijten et al. Fig. 2, Sec. III-A/B, Eq. 14):
    FC(N -> N/4) -> AntiRectifier -> Dropout
    FC(2N/4 -> N/4) -> AntiRectifier -> Dropout
    FC(2N/4 -> N/4) -> AntiRectifier -> Dropout
    FC(2N/4 -> N)   -> output weights

The AntiRectifier doubles width after each hidden layer (hence 2*h inputs
for layers 2-4), creating an encoder-decoder bottleneck at N/4 that forces
a compact representation and regularises the predicted weights.

Operates pixel-wise: input is [B*P, N] where B = batch and P = nx*nz.
"""
import torch
import torch.nn as nn


class AntiRectifier(nn.Module):
    """g(x) = [relu(x_hat), relu(-x_hat)], x_hat = (x - mean) / ||x - mean||_2

    Eq. 14 of Luijten et al.  Doubles feature width while preserving the
    sign of bipolar RF-derived signals that plain ReLU would suppress.
    """
    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=-1)


class ABLEMLP(nn.Module):
    """4-layer FC encoder-decoder with AntiRectifier activations.

    N   = M*M = number of channel pairs (e.g. 4096 for M=64)
    N/4 = bottleneck width (1024 for M=64)
    """
    def __init__(self, N, dropout=0.2):
        super().__init__()
        h = max(N // 4, 1)
        self.fc1 = nn.Linear(N,     h)       # encoder:  N   -> N/4
        self.fc2 = nn.Linear(2 * h, h)       #           2N/4 -> N/4
        self.fc3 = nn.Linear(2 * h, h)       #           2N/4 -> N/4
        self.fc4 = nn.Linear(2 * h, N)       # decoder:  2N/4 -> N
        self.act  = AntiRectifier()
        self.drop = nn.Dropout(dropout)

    def forward(self, y):
        """y: [B*P, N]  ->  w_rx: [B*P, N]  (apodization weights per pixel)"""
        x = self.drop(self.act(self.fc1(y)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.drop(self.act(self.fc3(x)))
        x = self.fc4(x)
        # Softmax to ensure: all weights > 0, sum = 1.0 per pixel
        return torch.softmax(x, dim=-1)


def beamform_weighted(pre_summed_pixels, weights):
    """Weighted sum of per-channel-pair contributions  (Eq. 13 generalised).

    pre_summed_pixels: [B*P, M*M]   aligned channel-pair values (MLP input)
    weights:           [B*P, M*M]   apodization weights (MLP output)
    returns:           [B*P]        reconstructed pixel amplitude
    """
    return (weights * pre_summed_pixels).sum(dim=-1)


def apply_mlp(mlp, pre_summed):
    """Full pixel-wise apodization pass, handling the reshape in one place.

    pre_summed: [B, M*M, P]
    returns:
        p_recon  [B, P]       reconstructed image
        weights  [B*P, M*M]   predicted weights (needed for unity-gain loss)
        x        [B*P, M*M]   reshaped pre_summed (reused in loss / beamform)
    """
    B, MM, P = pre_summed.shape
    x       = pre_summed.permute(0, 2, 1).reshape(B * P, MM)   # [B*P, M*M]
    weights = mlp(x)                                            # [B*P, M*M]
    p_recon = beamform_weighted(x, weights).reshape(B, P)       # [B, P]
    return p_recon, weights, x
