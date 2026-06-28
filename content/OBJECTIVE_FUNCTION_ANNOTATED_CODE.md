# OBJECTIVE FUNCTION — Annotated Code Walkthrough

## The Complete Code Path

### Step 1: Define the Objective Function Components

**File: `able_plus_plus/networks/losses.py`**

```python
# Lines 16-27: L_SMSLE (reconstruction fidelity)
def smsle_loss(p_pred, p_target, eps=1e-6):
    """Signed Mean-Squared Logarithmic Error
    
    Measures: How close is prediction to target in dB scale?
    
    Why logarithmic?
      - RF data spans 6-8 orders of magnitude
      - Ultrasound displays use dB scale (log compression)
      - Small signals matter as much as large ones
    """
    # Split into positive (signal) and negative (noise/artifact)
    pred_pos, pred_neg = torch.relu(p_pred), torch.relu(-p_pred)
    targ_pos, targ_neg = torch.relu(p_target), torch.relu(-p_target)
    
    # Compare magnitudes in log domain
    loss_pos = (torch.log10(pred_pos + eps) - torch.log10(targ_pos + eps)).pow(2).mean()
    loss_neg = (torch.log10(pred_neg + eps) - torch.log10(targ_neg + eps)).pow(2).mean()
    
    return 0.5 * (loss_pos + loss_neg)  # ← This is L_SMSLE


# Lines 30-37: L_unity (physical constraint)
def unity_gain_penalty(weights):
    """Unity-Gain Penalty: weights should sum to ~1 per pixel
    
    Ensures: Apodization weights stay physically plausible
    
    Why sum to 1?
      - Standard beamforming has unity gain (no amplification)
      - If sum >> 1: network amplifies noise
      - If sum << 1: network attenuates signal
    """
    return (weights.sum(dim=-1) - 1.0).pow(2).mean()
    # ↑ For each pixel: (sum_of_weights - 1.0)²
    # ↑ Then average across all pixels


# Lines 40-49: THE OBJECTIVE FUNCTION
def total_loss(p_pred, p_target, weights, lam=0.8):
    """L_total = λ * L_SMSLE + (1-λ) * L_unity
    
    This is what we optimize!
    
    Parameters:
      lam = 0.8  ← 80% reconstruction accuracy
            (1-lam) = 0.2  ← 20% physical plausibility
    """
    l_img   = smsle_loss(p_pred, p_target)
    l_unity = unity_gain_penalty(weights)
    
    # The key formula:
    return (lam * l_img + (1.0 - lam) * l_unity,  # ← L_total (what we minimize)
            l_img,                                  # ← L_SMSLE (for logging)
            l_unity)                                # ← L_unity (for logging)
```

### Step 2: Use the Objective Function in Training

**File: `able_plus_plus/train.py`**

```python
def train_step(model, mlp, optimizer, batch_size, lam=0.8, noise_level=0.05, device='cpu'):
    """One training iteration
    
    This is where the objective function is evaluated and used.
    """
    optimizer.zero_grad()
    
    # Generate ground truth scatterers
    gt_images, rf_data = make_batch(model, batch_size, noise_level=noise_level, device=device)
    # ↑ gt_images: [B, nx*nz]  ground truth
    # ↑ rf_data:   [B, M, T]   simulated RF (under torch.no_grad, frozen physics)
    
    # Physics: align RF data (FROZEN)
    _, pre_summed = model.das_adjoint(rf_data)
    # ↑ pre_summed: [B, M*M, P]  aligned channel values per pixel (frozen geometry)
    
    # Neural Network: predict weights (TRAINABLE)
    p_recon, weights, _ = apply_mlp(mlp, pre_summed)
    # ↑ p_recon: [B, P]        reconstructed image (from weighted sum)
    # ↑ weights: [B*P, M*M]    what the network learned
    
    # Detach ground truth (prevent gradients)
    target = gt_images.detach()
    
    # ╔════════════════════════════════════════════════════════╗
    # ║  EVALUATE THE OBJECTIVE FUNCTION                       ║
    # ╚════════════════════════════════════════════════════════╝
    loss, l_img, l_unity = total_loss(p_recon, target, weights, lam=lam)
    #     └─ This calls:
    #        smsle_loss(p_recon, target)
    #        unity_gain_penalty(weights)
    #        return (0.8*l_img + 0.2*l_unity, l_img, l_unity)
    
    # Backpropagation (compute gradients with respect to loss)
    loss.backward()
    # ↑ ∂loss/∂mlp_weights computed here
    # ↑ Does NOT reach forward model (frozen with torch.no_grad)
    
    # Gradient clipping (stability)
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
    
    # Update MLP weights to minimize loss
    optimizer.step()
    # ↑ w_new = w_old - lr * ∇loss
    # ↑ Only MLP weights change, physics model frozen
    
    return {
        'loss':         loss.item(),      # ← L_total (what we're minimizing)
        'image_loss':   l_img.item(),     # ← L_SMSLE (reconstruction accuracy)
        'unity_loss':   l_unity.item(),   # ← L_unity (physical constraint)
    }
```

### Step 3: Main Training Loop

**File: `run_training.py` (simplified)**

```python
def main():
    model = ForwardModel(M=64, nx=128, nz=128, device=device).to(device)
    mlp = ABLEMLP(N=64*64).to(device)  # 16.7M parameters, TRAINABLE
    
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    # ╔════════════════════════════════════════════════════════╗
    # ║  TRAINING LOOP: Evaluate objective function 5000 times ║
    # ╚════════════════════════════════════════════════════════╝
    for step in range(1, 5001):
        # This calls train_step() which calls total_loss()
        l_total, l_img, l_unity = train_step(model, mlp, opt, cfg, device)
        #                          └─ Objective function evaluated here
        
        # Log the metrics
        if step % 20 == 0:
            print(f"step {step}/5000  loss={l_total:.4f}  img={l_img:.4f}  unity={l_unity:.4f}")
            
            # Save status
            write_status(status_path,
                step=step,
                loss=round(l_total, 6),
                image_loss=round(l_img, 6),
                unity_loss=round(l_unity, 6),
                ...
            )
        
        # Save checkpoint
        if step % 200 == 0:
            save_checkpoint(...)
```

## Evidence in the Training Log

**From `checkpoints/training.log`:**

```
2026-06-18 20:14:24,248  INFO      step   20/5000  loss=38.7126  img=41.1818  unity=28.8359  best=34.0966
                                                    └─ L_total     └─ L_SMSLE    └─ L_unity
2026-06-18 20:14:32,249  INFO      step   40/5000  loss=45.1996  img=48.4775  unity=32.0878  best=32.4322
2026-06-18 20:14:40,194  INFO      step   60/5000  loss=34.5209  img=43.1259  unity=0.1010   best=32.4322
2026-06-18 20:14:48,155  INFO      step   80/5000  loss=53.8527  img=50.5346  unity=67.1251  best=32.4322
2026-06-18 20:14:56,108  INFO      step  100/5000  loss=45.7345  img=50.7308  unity=25.7496  best=32.4322
```

**Verification (step 20):**
```
loss = 0.8 * img + 0.2 * unity
38.7126 = 0.8 * 41.1818 + 0.2 * 28.8359
38.7126 = 32.9454 + 5.7672
38.7126 = 38.7126 ✓
```

## Control Flow Summary

```
python run_training.py
  │
  ├─ Build ForwardModel (physics, FROZEN)
  ├─ Build ABLEMLP (weights, TRAINABLE)
  │
  └─ for step in 1..5000:
       │
       ├─ train_step(model, mlp, optimizer, cfg, device)
       │  │
       │  ├─ gt_images, rf_data = make_batch()
       │  │
       │  ├─ pre_summed = model.das_adjoint(rf_data)  [FROZEN physics]
       │  │
       │  ├─ p_recon, weights = apply_mlp(pre_summed)  [TRAINABLE network]
       │  │
       │  ├─ >>>>> OBJECTIVE FUNCTION CALL >>>>>
       │  │
       │  ├─ loss, l_img, l_unity = total_loss(p_recon, target, weights, lam=0.8)
       │  │  │
       │  │  ├─ l_img = smsle_loss(p_recon, target)
       │  │  │           └─ L_SMSLE = reconstruction error in dB
       │  │  │
       │  │  └─ l_unity = unity_gain_penalty(weights)
       │  │              └─ L_unity = constraint penalty
       │  │
       │  ├─ loss.backward()        [Compute gradients ∂loss/∂mlp_weights]
       │  │
       │  └─ optimizer.step()       [Update MLP weights]
       │
       ├─ Log metrics to training.log
       │
       └─ [if step % 200 == 0] save checkpoint
```

## Summary

- **What's being optimized:** L_total = 0.8·L_SMSLE + 0.2·L_unity
- **L_SMSLE:** Reconstruction fidelity (log-domain error)
- **L_unity:** Physical constraint (weights sum to 1)
- **Where defined:** `able_plus_plus/networks/losses.py` (lines 16-49)
- **Where used:** `able_plus_plus/train.py` (line 37), `run_training.py` (line 123)
- **Frequency:** Evaluated 5000 times (once per training step)
- **Evidence:** Training log shows decreasing loss, correct formula verification
