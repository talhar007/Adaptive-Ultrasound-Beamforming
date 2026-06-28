# Bug Fix Report: MLP Output Constraint

## 🐛 The Bug

The MLP output layer had **no activation function**, causing:

1. **Negative weights** (ranging from -0.18 to 0.19)
2. **Weights don't sum to 1.0** (sum to only ~0.848)
3. **Signal attenuation** instead of proper beamforming
4. **Noisy visual reconstruction** despite low MAE

### Example
```
Bad:  weights = [-0.05, 0.12, 0.08, 0.10, ...]
      sum(weights) = 0.848  ← Attenuates signal!
      Negative weight confuses beamforming
```

---

## 🔧 The Fix

Added **softmax** to the MLP output:

**Before (WRONG):**
```python
def forward(self, y):
    x = self.drop(self.act(self.fc1(y)))
    x = self.drop(self.act(self.fc2(x)))
    x = self.drop(self.act(self.fc3(x)))
    return self.fc4(x)  # ← NO activation! Bad!
```

**After (CORRECT):**
```python
def forward(self, y):
    x = self.drop(self.act(self.fc1(y)))
    x = self.drop(self.act(self.fc2(x)))
    x = self.drop(self.act(self.fc3(x)))
    x = self.fc4(x)
    return torch.softmax(x, dim=-1)  # ← Softmax! Correct!
```

---

## ✅ What Softmax Does

Softmax ensures:
- ✅ All weights are **positive** (0 to 1)
- ✅ Weights **sum to 1.0** per pixel (automatic!)
- ✅ **No attenuation** (proper beamforming)
- ✅ **Physical meaning** (valid apodization weights)

### Example with softmax
```
Good:  raw = [-1.5, 0.8, 0.2, 1.0, ...]
       softmax(raw) = [0.02, 0.24, 0.15, 0.28, ...]
       sum = 1.0  ← Perfect!
       All weights > 0  ← Good!
```

---

## 📊 Impact

**Before fix:**
- Weights: -0.18 to +0.19, sum ≈ 0.848
- Output: Noisy reconstruction
- Visual quality: Worse than DAS
- MAE: Low (misleading!)

**After fix (expected):**
- Weights: 0 to 1, sum = 1.0 exactly
- Output: Sharp dots like ground truth
- Visual quality: Better than DAS
- MAE: Lower AND visually sharper

---

## 🚀 Next Steps

1. **Delete old checkpoint** (has wrong weights)
   ```bash
   rm checkpoints/checkpoint_latest.pt
   ```

2. **Retrain from scratch**
   ```bash
   python scripts/run_training.py --n_steps 5000 --batch_size 4
   ```

3. **Test again**
   ```bash
   python scripts/demo_results.py
   ```

4. **Verify results**
   - Check case_0X_reconstruction.png
   - ABLE should now show sharp dots, not noise
   - Visual quality should match/beat DAS
   - MAE should still be low

---

## Why This Matters

The **unity constraint (L_unity) is essential**:
- Without softmax: L_unity tries to constrain unbounded weights (ineffective)
- With softmax: Weights are automatically bounded and sum to 1.0 (optimal)

This is a critical fix for the algorithm to work correctly.

---

**File modified:** `able_plus_plus/networks/able_mlp.py` line 59

**Status:** ✅ Fixed and ready to retrain
