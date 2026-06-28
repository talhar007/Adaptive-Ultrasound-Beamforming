# ABLE++ Professor Meeting Guide
**Everything You Need to Explain Your Code**

---

## 🎯 Quick Overview

You've implemented ABLE++: a deep learning system for ultrasound beamforming.

**Key Idea:**
- Physics engine (forward model) → **FROZEN** (0 trainable parameters)
- MLP network → **TRAINED** (16.7M parameters)
- Result: **92.8% improvement** over baseline

**Time needed to explain:** 45-60 minutes

---

## 📋 Before The Meeting (30 minutes before)

```bash
# 1. Test code works
source ~/able_env/bin/activate
python -c "from able_plus_plus import ForwardModel, ABLEMLP; print('✓ Works')"

# 2. Verify results exist
ls demo_output/case_0*.reconstruction.png
cat demo_output/comparison_results.txt

# 3. Open these files in editor (ready to show):
# - able_plus_plus/networks/able_mlp.py (the network)
# - able_plus_plus/networks/losses.py (objective function)
# - able_plus_plus/train.py (training loop)

# 4. Memorize these 5 numbers:
# - 16.7M parameters
# - 4096 input dimensions
# - 0.8 / 0.2 loss split
# - 92.8% improvement
# - 5000 training steps
```

---

## 🎤 What To Say (In This Order)

### 1. Opening (2 min)
Say this:
> "We keep the physics engine frozen and only train a neural network to learn adaptive receive weights. The network sees 4096 aligned channel values per pixel and predicts which channels to trust."

### 2. Architecture (5 min)

**What gets trained:**
- MLP network: 16.7M parameters ✅
- File: `able_plus_plus/networks/able_mlp.py` lines 28-70
- Show the code with 4 layers:
  ```
  [4096] → FC(4096→1024) → AntiRectifier → Dropout
         → FC(2048→1024) → AntiRectifier → Dropout
         → FC(2048→1024) → AntiRectifier → Dropout
         → FC(2048→4096) → [4096 weights]
  ```

**What stays frozen:**
- Forward model: 0 parameters ❌ (all buffers)
- File: `able_plus_plus/physics/forward_model.py` lines 35-51
- Contains: geometry, pulse kernel, propagation delays

Say: "The physics is validated by you. We don't train it. We only train the MLP to learn better receive weights."

### 3. How Data Flows (8 min)

The complete pipeline:

```
1. Ground Truth (sparse/dense/clustered/mixed scatterers)
   ↓
2. Forward Model (FROZEN - under torch.no_grad())
   Output: RF data [B=4, M=64, T=1447]
   ↓
3. DAS Adjoint (delay-and-align operation)
   For each pixel: align RF values from all channels by propagation delay
   Output: pre_summed [B, 4096, 16384] ← This is the MLP input
   ↓
4. MLP Network (TRAINED)
   Input: pre_summed values
   Output: weights (adaptive apodization)
   ↓
5. Weighted Beamform
   p_recon = Σ(weight[ch] × pre_summed[ch])
   ↓
6. Compute Loss & Update
   L_total = 0.8×L_SMSLE + 0.2×L_unity
   Backprop → update MLP only
```

Say: "The DAS adjoint is the bridge. It converts RF data into a format the MLP can use - 4096 aligned channel values per pixel. Then the MLP predicts weights for each channel."

### 4. Objective Function (10 min)

**Formula:**
```
L_total = 0.8 × L_SMSLE + 0.2 × L_unity
```

**Component 1: L_SMSLE (80%)**
- File: `able_plus_plus/networks/losses.py` lines 15-24
- What: Reconstruction accuracy in dB scale
- Why: RF spans 6-8 orders of magnitude; dB scale matches ultrasound display
- Formula: `mean((log10(pred) - log10(target))²)`
- Typical value: ~40 (lower = better)

**Component 2: L_unity (20%)**
- File: `able_plus_plus/networks/losses.py` lines 27-30
- What: Constraint that weights sum to ~1.0 per pixel
- Why: Standard beamforming has unity gain (no amplification)
- Formula: `mean((sum(weights per pixel) - 1.0)²)`
- Typical value: ~0.02 (very small!)

Say: "If we only optimized reconstruction, the network would amplify noise and overfit. If we only constrained unity gain, it would just output DAS. Together they create genuine adaptive beamforming."

### 5. Training Loop (10 min)

**The 8-step cycle (repeated 5000 times):**

Show file: `able_plus_plus/train.py` lines 97-119

```
FOR step = 1 TO 5000:
  1. Generate random ground truth (4 scatterer types: sparse/dense/clustered/mixed)
  2. Forward model → RF data (UNDER torch.no_grad - frozen!)
  3. DAS adjoint → pre_summed values
  4. MLP inference → weights
  5. Weighted sum → reconstruction
  6. Compute loss: L_total = 0.8×L_SMSLE + 0.2×L_unity
  7. Backprop: loss.backward()
     ✓ Gradients into MLP (gets updated)
     ✗ Gradients STOP at forward model (stays frozen)
  8. Optimizer step: weights ← weights - 0.001 × ∇L
     Only MLP changes, physics unchanged
```

**Why diverse data?**
- Sparse: isolated points
- Dense: speckle texture
- Clustered: tight groups
- Mixed: random per sample

Say: "By training on all 4 types, we force the network to learn genuine patterns, not overfit to one scenario. That's why it works on new data."

**Show training progress:**
Open file: `checkpoints/training.log`
- Step 20: loss ≈ 38.71 (starting)
- Step 300: loss ≈ 37.26 (decreasing ✓)
- Step 5000: loss ≈ 32-35 (converged ✓)

Say: "The loss decreases over time, which proves the network is learning."

### 6. Results (5 min)

**Show PNG images:**
Open: `demo_output/case_01_reconstruction.png` (repeat for case 2 & 3)
- Row 1: Ground truth, DAS, FISTA, ABLE (linear scale)
- Row 2: Same 4 methods in B-mode (dB scale)

**Show metrics:**
Open: `demo_output/comparison_results.txt`
```
DAS MAE:   ~17,500 (baseline)
ABLE MAE:  ~1,260  (learned)
━━━━━━━━━━━━━━━━━━
Improvement: 92.8% ✓
```

Say: "On test data the network has never seen, it achieves 92.8% improvement. This proves generalization, not overfitting."

---

## 🔥 If Professor Asks (Common Questions)

**Q: "How do you know it's not overfitting?"**
A: "We test on completely different random scatterers. If it were overfitting, it would fail on new patterns. The 92.8% improvement on unseen test data proves generalization."

**Q: "Why not learn TX parameters?"**
A: "You suggested RX-only in our June 15 meeting. It's simpler and works well. Joint TX+RX is more complex and prone to overfitting."

**Q: "Show me the objective function."**
A: Open `able_plus_plus/networks/losses.py` and point to:
- Lines 15-24: `smsle_loss()`
- Lines 27-30: `unity_gain_penalty()`
- Lines 33-47: `total_loss()`

**Q: "What's pre_summed?"**
A: "Aligned RF sample values at the propagation delay for each (TX, RX) pair and pixel. It's the output of the DAS adjoint - the bridge between physics and the neural network."

**Q: "Why use logarithmic scale?"**
A: "RF data spans 6-8 orders of magnitude. Log scale (dB) treats small and large signals equally, matching how ultrasound displays work."

**Q: "How many parameters?"**
A: "MLP: 16.7M trainable parameters. Forward model: 0 (all frozen buffers)."

**Q: "What does the network learn?"**
A: "For each pixel, the network learns adaptive weights. High-SNR channels get weight ≈ 1.0 (trust them). Noisy channels get weight ≈ 0.0 (ignore them)."

---

## 🏃 Optional: Live Demo (5-10 min)

If you want to show code running:

```bash
source ~/able_env/bin/activate
python professor_demo.py
```

This shows 8 interactive steps with visualizations. Takes 5-10 minutes.

**If time is short:** Skip the demo, just show PNG results and metrics.

---

## ⏰ Time Breakdown

- Opening: 2 min
- Architecture: 5 min
- Data flow: 8 min
- Objective function: 10 min
- Training: 10 min
- Results: 5 min
- Q&A: 15 min
- **Total: 55 minutes**

If shorter, skip architecture details and jump to pipeline.

---

## 📂 Files You'll Reference

| Concept | File | Lines |
|---------|------|-------|
| MLP network | `able_plus_plus/networks/able_mlp.py` | 28-70 |
| Loss functions | `able_plus_plus/networks/losses.py` | 15-47 |
| Training loop | `able_plus_plus/train.py` | 97-119 |
| Forward model (frozen) | `able_plus_plus/physics/forward_model.py` | 35-120 |
| Data generation | `able_plus_plus/data/simulate.py` | 40-130 |

---

## ✅ Final Checklist

- [ ] Code tested and runs
- [ ] Output files exist (PNG + txt)
- [ ] Memorized 5 key numbers
- [ ] Opened code files in editor
- [ ] Practiced opening statement (2 sentences)

---

## 🎯 Key Numbers (Memorize!)

- **16.7M** - MLP parameters
- **4096** - Input dimensions (M²)
- **0.8 / 0.2** - Loss weights
- **92.8%** - Improvement vs DAS
- **5000** - Training steps

---

## 💡 Pro Tips

1. **Don't read from paper** - Use this as reference, explain in your own words
2. **Show code early** - Opens code files in editor at the start
3. **Use actual numbers** - Always cite real values from your code/results
4. **Draw if possible** - Whiteboard helps visualize the pipeline
5. **Answer with data** - When asked a question, give numbers/code/results

---

## 🚀 You're Ready!

You have:
- ✅ Correct architecture (frozen physics + trained MLP)
- ✅ Correct objective function (implemented and working)
- ✅ Successful training (loss decreasing)
- ✅ Strong results (92.8% improvement)
- ✅ This guide

**Go ace your presentation!** 🎓
