"""
VISUAL PIPELINE WALKTHROUGH
===========================

This document shows step-by-step what happens during training,
with tensor shapes and which components are frozen vs trainable.
"""

STEP-BY-STEP TRAINING PIPELINE
===============================

STEP 1: Generate Ground Truth
─────────────────────────────
  
  random_point_scatterers(B=4, nx=128, nz=128)
  
  Output:
    gt_images [4, 16384]  = ground truth reflectivity
    
  Visual (2D):
    ╔────────────╗
    │ █ █        │  Point scatterers:
    │  █  █      │  - sparse (2-6 per batch)
    │    █ █     │  - dense (many overlapping)
    │ █  █       │  - clustered (tight groups)
    │     █      │  
    ╚────────────╝  → Used as optimization target


STEP 2: Forward Model (FROZEN — Physics Engine)
────────────────────────────────────────────────

  model.simulate(gt_images)
    Input:  gt_images [4, 16384]  ground truth scatterers
    Output: rf_data [4, 64, 1447]  received RF signals
  
  What happens inside:
    ┌─────────────────────────────────────────┐
    │  For each scatterer location:            │
    │    1. Spread to time samples (delay)     │  [FIXED geometry]
    │    2. Convolve with broadband pulse      │  [FIXED pulse]
    │    3. Sum over all TX transmitters       │
    │  Result: RF at each of 64 RX elements    │
    └─────────────────────────────────────────┘
  
  Status: ✗ NOT TRAINABLE (torch.no_grad())
          ✓ Frozen geometry buffers
          ✓ Frozen pulse kernel


STEP 3: Add Noise (Realistic Data)
──────────────────────────────────

  noise = randn_like(rf) * 0.05 * rf.abs().max()
  rf_noisy = rf + noise
  
  Output: rf_noisy [4, 64, 1447]
  
  Status: ✓ Creates realistic training data
          ✓ Prevents overfitting to clean signals


STEP 4: DAS Adjoint (Delay & Align) — FROZEN
──────────────────────────────────────────────

  das_image, pre_summed = model.das_adjoint(rf_noisy)
  
  Input:  rf_noisy [4, 64, 1447]
  Output:
    - das_image [4, 16384]      uniform sum → DAS baseline
    - pre_summed [4, 4096, 16384] per-channel contributions before sum
  
  What happens inside:
    ┌────────────────────────────────────────────┐
    │  For each (tx, rx, pixel):                 │
    │    1. Look up propagation delay (distance)  │ [FIXED buffers]
    │    2. Pick RF sample at that delay          │
    │    3. Interpolate fractionally              │
    │    4. Scale by 1/distance (amplitude decay) │ [FIXED gains]
    │  Result: aligned channel value per pixel    │
    │                                             │
    │  Stack as [4, 4096, 16384]:                │
    │    dim 0: batch (4 images)                  │
    │    dim 1: channel pairs (64*64 = 4096)      │
    │    dim 2: pixels (128*128 = 16384)          │
    └────────────────────────────────────────────┘
  
  Status: ✓ Autograd enabled (for gradient flow later)
          ✓ Geometry frozen (uses buffers, no parameters)
          ✓ This is the bridge between physics and neural network


STEP 5: Reshape for MLP (Pixel-wise Processing)
────────────────────────────────────────────────

  pre_summed [4, 4096, 16384]
    ↓ permute(0, 2, 1)
  [4, 16384, 4096]
    ↓ reshape(4*16384, 4096)
  x [65536, 4096]
  
  Interpretation:
    Each row = one pixel's 4096 aligned channel values
    [pixel_0_ch0, pixel_0_ch1, ..., pixel_0_ch4095]
    [pixel_1_ch0, pixel_1_ch1, ..., pixel_1_ch4095]
    ...
    [pixel_65535_ch0, ..., pixel_65535_ch4095]
  
  Status: ✓ Ready for pixel-wise MLP processing


STEP 6: MLP Forward Pass (TRAINABLE)
─────────────────────────────────────

  weights = mlp(x)
  
  Input:  x [65536, 4096]  per-pixel channel values
  Output: weights [65536, 4096]  per-pixel apodization weights
  
  Network architecture:
    [65536, 4096]
         ↓
    FC(4096 → 1024) + AntiRectifier + Dropout
         ↓
    [65536, 2048]  ← doubled by AntiRectifier
         ↓
    FC(2048 → 1024) + AntiRectifier + Dropout
         ↓
    [65536, 2048]
         ↓
    FC(2048 → 1024) + AntiRectifier + Dropout
         ↓
    [65536, 2048]
         ↓
    FC(2048 → 4096)  ← output layer
         ↓
    [65536, 4096]  ← one weight per channel per pixel
  
  Status: ✓ TRAINABLE (gradient backprop enabled)
          ✓ Parameters: 16,784,384 (16.7M)
          ✓ Learned through backpropagation


STEP 7: Beamform with Learned Weights (Weighted Sum)
─────────────────────────────────────────────────────

  Reconstructed image = (weights * x).sum(dim=-1)
  
  Input:  weights [65536, 4096]  learned apodization
          x [65536, 4096]        aligned channels (from Step 5)
  
  Operation (per pixel):
    pixel_value = w[0]*ch[0] + w[1]*ch[1] + ... + w[4095]*ch[4095]
    
  Output: p_recon [4, 16384]  reconstructed image
  
  Visual:
    ╔──────────────────────────────────╗
    │ Element 0: w[0]=0.8  ch[0]=50    │
    │ Element 1: w[1]=0.6  ch[1]=30    │
    │ Element 2: w[2]=0.1  ch[2]=5     │  Pixel value =
    │ Element 3: w[3]=0.2  ch[3]=10    │  weighted sum
    │ ...                              │  = 0.8*50 + 0.6*30 + 0.1*5 + ...
    │ Element 4095: w[4095]=0.05 ch=2  │
    ╚──────────────────────────────────╝
  
  Status: ✓ Differentiable (weights carry gradients)
          ✓ This is where the network's learning is applied


STEP 8: Compute Objective Function
───────────────────────────────────

  L_SMSLE = smsle_loss(p_recon, gt)
  L_unity = unity_gain_penalty(weights)
  L_total = 0.8 * L_SMSLE + 0.2 * L_unity
  
  ╔════════════════════════════════════════════════════╗
  ║  OBJECTIVE FUNCTION (What we optimize for)         ║
  ╚════════════════════════════════════════════════════╝
  
  L_SMSLE Computation:
    ┌─────────────────────────────────────────────────┐
    │ Split into positive and negative:                │
    │   pred_pos = relu(p_recon)    [amplitudes ≥ 0]  │
    │   pred_neg = relu(-p_recon)   [amplitudes ≤ 0]  │
    │   targ_pos = relu(gt)                            │
    │   targ_neg = relu(-gt)                           │
    │                                                  │
    │ Compare in log domain:                           │
    │   loss_pos = mean((log10(pred_pos) - log10(...))²) │
    │   loss_neg = mean((log10(pred_neg) - log10(...))²) │
    │                                                  │
    │ L_SMSLE = 0.5 * (loss_pos + loss_neg)            │
    └─────────────────────────────────────────────────┘
    
    Meaning: "How far off in dB (perceptual ultrasound space)?"
    Scalar: e.g., 41.18
  
  L_unity Computation:
    ┌─────────────────────────────────────────────────┐
    │ For each pixel, check weight sum:                │
    │   sum_per_pixel = weights.sum(dim=-1) [65536]   │
    │                                                  │
    │ Penalize if not ~1:                             │
    │   L_unity = mean((sum_per_pixel - 1.0)²)        │
    │                                                  │
    │ Example:                                         │
    │   Pixel A: weights sum to 0.5  → penalty = 0.25 │
    │   Pixel B: weights sum to 1.2  → penalty = 0.04 │
    │   Pixel C: weights sum to 1.0  → penalty = 0.00 │
    │   L_unity = mean([0.25, 0.04, 0.00, ...])      │
    └─────────────────────────────────────────────────┘
    
    Meaning: "Do the weights stay physically plausible?"
    Scalar: e.g., 28.84
  
  L_total = 0.8 * 41.18 + 0.2 * 28.84 = 38.71
    80% cares about reconstruction accuracy
    20% cares about realistic weights


STEP 9: Backpropagation (Gradient Computation)
───────────────────────────────────────────────

  loss.backward()
  
  Gradient flow:
  
    L_total [scalar]
       ↓
    ∂L/∂weights [65536, 4096]  ← MLP weights get gradients ✓
       ↓
    ∂L/∂hidden_layers
       ↓
    ∂L/∂MLP_parameters [16.7M values]
    
    NOT propagated:
    ✗ ∂L/∂forward_model (deliberately frozen)
    ✗ ∂L/∂geometry_buffers (not parameters, frozen)
    ✗ ∂L/∂pulse (not parameters, frozen)
  
  Status: ✓ Gradients computed for MLP only
          ✓ Forward model remains unchanged


STEP 10: Optimizer Step (Parameter Update)
───────────────────────────────────────────

  optimizer.step()
  
  Update rule (Adam optimizer):
    new_param = old_param - lr * gradient  (simplified)
    
  In our case:
    new_mlp_weights = old_mlp_weights - 1e-3 * (∂L/∂mlp_weights)
    
  Effect: MLP weights shift in direction that reduces loss
  
  Status: ✓ Only MLP weights change
          ✓ Forward model unchanged
          ✓ Next iteration uses updated MLP


LOOP 5000 TIMES
═══════════════

  Step 1-2: Generate new random scatterers (different each time)
  Step 3-10: Train on the loss
  
  Iteration 1:    L_total = 38.71
  Iteration 2:    L_total = 45.20
  Iteration 3:    L_total = 34.52
  ...
  Iteration 300:  L_total = 37.26
  ...
  Iteration 5000: L_total ≈ 25-30 (lower = better)
  
  Result: MLP learns to predict weights that reconstruct
          accurately while staying physically plausible


SUMMARY TABLE: What's Trainable vs Fixed?
══════════════════════════════════════════

  Component                          Trainable?   Status
  ───────────────────────────────────────────────────────
  Forward model (simulate)           ✗ NO         torch.no_grad()
  Pulse kernel                       ✗ NO         Buffer
  Array geometry (distances)         ✗ NO         Buffers
  DAS adjoint (das_adjoint)          ✗ NO         Frozen buffers
  MLP weights (fc1, fc2, fc3, fc4)   ✓ YES        nn.Parameters
  MLP biases                         ✓ YES        nn.Parameters
  ───────────────────────────────────────────────────────
  
  Why this design?
    - Physics is correct (validated by supervisor)
    - Only data-dependent part needs to adapt (apodization weights)
    - Keeps learning focused and interpretable
    - Prevents overfitting to the simulator


WHAT THE MLP LEARNS
═══════════════════

After 5000 training steps, the MLP learns:

  For scatterer pixel at (x=50, z=75):
    "High-valued channels (SNR>10): weights ≈ 1.0"
    "Low-valued channels (noise): weights ≈ 0.0"
    "Medium channels: weights ≈ 0.5"
  
  For noise-only pixel at (x=10, z=20):
    "All channels noisy: uniform low weights ≈ 0.1"
  
  For edge pixel (between strong and weak regions):
    "Some channels point to strong region: high weight"
    "Other channels point to weak region: low weight"

This adaptive, per-pixel, per-channel weighting is what
DAS cannot do (DAS uses same weights for all pixels/channels).
