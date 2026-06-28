# Professor Meeting Context — Ultrasound Beamforming Project (S6.EMS.O1)

## Key Clarification from Professor Meeting

The pipeline does **not** feed learned weights back into the forward model.
The forward model is a **fixed data generator only** — no gradients flow through it.

### Corrected Pipeline

```
POINT SCATTERERS (ground truth scene)
[sparse 1.0 values on a 128×128 blank grid — known locations]
        ↓
Forward Model (FIXED — data generator only)
feeds ground truth scatterer locations + amplitudes into physics simulation
outputs realistic RF channel data
        ↓
RF Data [Batch, M=64 channels, N_t=1447 time samples]
        ↓
Adjoint / DAS align (do NOT sum yet)
        ↓
Pre-summed samples [Batch, M*M=4096, nx*nz=16384]
(per-channel-pair aligned contributions, one vector per pixel)
        ↓
MLP (ABLE architecture) — THIS is what gets trained
Input:  per-pixel vector of 4096 aligned channel values
Output: per-pixel apodization weights (same size)
        ↓
Weighted sum → Reconstructed Image [Batch, nx*nz]
        ↓
Loss vs. ground truth point scatterer image
(SMSLE + design-specific constraint penalty)
        ↓
Backprop into MLP weights ONLY
```

---

## What the Forward Model Does (from forward_model_talha.ipynb)

- **Physics constants:** `c = 5920 m/s` (steel/NDT), `fc = 5 MHz`, `fs = 40 MHz`
- **Array:** `M = 64` elements, `pitch = 1mm`, imaging grid `128×128`
- **Pre-computes geometry** (in `torch.no_grad()`): transmit + receive distances, time delays in samples, linear interpolation weights, amplitude gain `1/total_dist` — all shape `[M*M, nx*nz]`
- **`forward_geom(images)`**: spreads scatterer amplitudes to correct time samples via `scatter_add_`
- **`forward_model(images, pulse_kernel)`**: convolves with broadband transducer pulse (grouped conv1d), sums over transmit elements → outputs RF data `[Batch, 64, 1447]`
- **`adjoint_model(rf_data)`**: DAS delay-and-align, returns:
  - `recon_image_flat` — standard DAS image `[Batch, nx*nz]`
  - `pre_summed_samples` — **the key tensor** `[Batch, M*M, nx*nz]` — per-channel contributions before final sum → this is the MLP input

---

## Where the MLP Plugs In

The `pre_summed_samples` tensor from `adjoint_model` is the **bridge between forward model and MLP**.

```python
# After adjoint_model:
_, pre_summed = engine.adjoint_model(rf_data)
# pre_summed: [Batch, 4096, 16384]

# Reshape for pixel-wise MLP input:
# [Batch*nx*nz, M*M] — each pixel gets its own channel vector
x = pre_summed.permute(0, 2, 1).reshape(B * nx * nz, M * M)

# MLP predicts weights per pixel:
weights = mlp(x)  # [Batch*nx*nz, M*M]

# Weighted sum to get reconstructed image:
x_samples = pre_summed.permute(0, 2, 1).reshape(B * nx * nz, M * M)
recon = (weights * x_samples).sum(dim=-1).reshape(B, nx * nz)
```

---

## What is Learned vs Fixed

| Component | Status |
|---|---|
| Forward model (physics) | **Fixed** — data generator only |
| Transmit delays `τ_tx` | Fixed (simultaneous transmit for now) |
| Transmit weights `w_tx` | Fixed for now |
| Pulse `p_0(t)` | Fixed |
| Receive apodization weights | **Learned by MLP** |

---

## Point Scatterers — Role in Training

- Ground truth = a few `1.0` values at known pixel locations on a blank grid
- Forward model simulates what the array physically receives from those scatterers
- MLP tries to reconstruct an image matching those exact point locations
- Good reconstruction = tight bright dots with no blur or sidelobes
- Loss measures how well learned weights recover the original point locations from noisy RF data

---

## Loss Function

```
L_total = λ · L_SMSLE + (1 - λ) · L_constraints
```

- `L_SMSLE`: signed mean-squared logarithmic error between reconstructed and ground truth image — main objective
- `L_constraints`: design-specific penalties (unity-gain on receive weights, regularization)
- Backprop updates **MLP weights only**

---

## MLP Architecture (ABLE from Luijten et al. IEEE TMI 2020)

- 4 fully-connected layers
- Outer layers: `N` nodes (= `M*M = 4096` or aperture size)
- Inner layers: `N/4` nodes (bottleneck encoder-decoder)
- Activation: **antirectifier** (not ReLU — RF data is bipolar, ReLU kills negatives)
- Dropout: 0.2 between every layer
- Operates **pixel-wise** — one forward pass per pixel
- Complexity: O(N²) vs MV beamforming O(N³)
