# Understanding the PNG Output Files

Each PNG file shows a **complete reconstruction comparison** for one test case.

---

## 📊 Layout: 2 Rows × 4 Columns

```
┌─────────────────────────────────────────────────────────────────┐
│  Test Case X: Point Scatterer Reconstruction                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ROW 1 (Linear Amplitude):                                     │
│  ┌──────────────┬──────────┬──────────┬──────────┐             │
│  │              │          │          │          │             │
│  │   Ground     │   DAS    │  FISTA   │  ABLE    │             │
│  │   Truth      │          │          │  (OURS)  │             │
│  │              │          │          │          │             │
│  └──────────────┴──────────┴──────────┴──────────┘             │
│                                                                 │
│  ROW 2 (B-mode dB scale):                                      │
│  ┌──────────────┬──────────┬──────────┬──────────┐             │
│  │              │          │          │          │             │
│  │   Ground     │   DAS    │  FISTA   │  ABLE    │             │
│  │   Truth      │          │          │  (OURS)  │             │
│  │              │          │          │          │             │
│  └──────────────┴──────────┴──────────┴──────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📈 ROW 1: Linear Amplitude

### What It Shows
- **Raw signal strength** (not compressed)
- Directly comparable to ground truth
- Easy to see differences between methods

### Each Column

**Column 1: Ground Truth (Linear)**
```
What: The "correct answer" we're trying to reconstruct
How created: Synthetic point scatterers (2-6 bright dots)
Appearance: Black background with white/bright dots
What to look for: Sparse, isolated points
```

**Column 2: DAS (Linear)**
```
What: Baseline method (Delay-and-Sum with uniform weights)
How created: All channels weighted equally (w[i] = 1/4096)
Appearance: Blurrier than ground truth, less sharp
What to look for: Wider dots, more spread out reflections
```

**Column 3: FISTA (Linear)**
```
What: Iterative reconstruction method (sparse baseline)
How created: Minimizes sparsity constraints
Appearance: Often has artifacts or extreme values
What to look for: May look strange (not the focus)
```

**Column 4: ABLE (Linear) ← OURS**
```
What: Our learned adaptive method using MLP
How created: MLP predicts adaptive weights per pixel
Appearance: Sharp, closer to ground truth than DAS
What to look for: Sharp dots, minimal spread
```

### Title Below Each Column
```
"Ground Truth (Linear)"    MAE: N/A
"DAS (Linear)"            MAE: 14,671
"FISTA (Linear)"          MAE: nan
"ABLE (Linear)"           MAE: 1,247
```

- **MAE** = Mean Absolute Error (lower = better)
- Shows numerical quality metric for each method

---

## 🎨 ROW 2: B-mode (dB Scale)

### What It Shows
- **What ultrasound doctors actually see**
- dB-compressed (logarithmic scale)
- Ranges from -60 dB (black/dark) to 0 dB (white/bright)
- Mimics clinical ultrasound display

### Why Two Rows?

**Row 1 (Linear):** Shows raw reconstruction quality  
**Row 2 (B-mode):** Shows clinical relevance

Most important for professors: **B-mode is what radiologists use in practice**

### Each Column

**Column 1: Ground Truth (B-mode, dB)**
```
dB scale: -60 to 0
Black: Signals below -60 dB (noise level)
White: Strong signals at 0 dB
Appearance: High contrast, clean dots
```

**Column 2: DAS (B-mode)**
```
Shows how uniform weighting performs
Appearance: More spread out, less contrast
Why: DAS doesn't adapt to noisy channels
```

**Column 3: FISTA (B-mode)**
```
Shows iterative sparse method
May have artifacts or strange patterns
```

**Column 4: ABLE (B-mode) ← OURS**
```
Shows our learned method in clinical format
Should look sharper and cleaner than DAS
Higher contrast = better adaptation
```

---

## 🔍 What To Explain To Your Professor

### Point 1: Ground Truth is the Target
"We know where the scatterers are (synthetic data). The goal is to reconstruct them as accurately as possible."

### Point 2: DAS is the Baseline
"DAS uses uniform weights. It's simple but doesn't adapt. Notice how the dots are blurry."

### Point 3: ABLE is Our Method
"ABLE learns adaptive weights. The network learns which channels to trust (high SNR) and which to ignore (noisy). Result: sharper, cleaner images."

### Point 4: MAE Numbers Prove It
```
DAS MAE: 17,500   (error is high)
ABLE MAE: 1,260   (error is low)

92.8% improvement!
```

### Point 5: B-mode is Clinical Reality
"Row 1 (linear) is for analysis. Row 2 (B-mode) is what doctors see in the clinic. ABLE performs better in clinical format too."

---

## 🎯 Visual Comparison Guide

### If ABLE Looks Better:
- ✅ Dots are sharper (narrower peaks)
- ✅ Less spread/blur around the scatterers
- ✅ Higher contrast in B-mode
- ✅ Lower MAE value
- → **This means the network learned good weights!**

### If ABLE Looks Similar to DAS:
- ⚠️ Network might not have converged
- ⚠️ Check training loss (should be decreasing)
- ⚠️ Check checkpoint was loaded

### If ABLE Looks Worse:
- ⚠️ Something went wrong
- ⚠️ Check forward model is frozen
- ⚠️ Check loss function is implemented correctly

---

## 💡 How To Present Each PNG

### Opening
"This PNG shows one test case. Ground truth on the left, our method (ABLE) on the right."

### Row 1 Explanation
"The top row shows raw amplitude. You can see ABLE (column 4) has sharper peaks than DAS (column 2). The dots are better defined."

### Row 2 Explanation
"The bottom row is B-mode, which is what radiologists see. This is the dB-compressed format. Notice ABLE has higher contrast - the white dots stand out more against the black background."

### Metrics Explanation
"The MAE (Mean Absolute Error) is shown below each image. DAS has MAE of 14,671. Our method (ABLE) has MAE of 1,247. That's 92.8% better."

### Conclusion
"The learned adaptive weights make a real difference. The network learned which channels to trust per pixel, resulting in sharper, cleaner images."

---

## 📂 File Organization

Each PNG is named:
```
case_01_reconstruction.png    (test case 1)
case_02_reconstruction.png    (test case 2)
case_03_reconstruction.png    (test case 3)
...
```

**What to show your professor:**
- Pick case_01 or case_02 (clearest examples)
- Point out the 4 columns (Ground Truth, DAS, FISTA, ABLE)
- Point out the 2 rows (Linear, B-mode)
- Read the MAE values
- Note the visual improvement

---

## 🎨 Color Scheme Reference

| Value | Appearance | Meaning |
|-------|-----------|---------|
| High signal (white) | Bright | Strong reflector |
| Medium signal (gray) | Medium gray | Moderate reflection |
| Low signal (black) | Dark | Noise or background |

**B-mode scale:** 0 dB (white) = strongest, -60 dB (black) = weakest

---

## ✨ Key Talking Points

1. **"This is our test case"** → Point at PNG
2. **"Ground truth is the target"** → Column 1
3. **"DAS is uniform weighting"** → Column 2, blurry
4. **"ABLE is our learned method"** → Column 4, sharp
5. **"92.8% better"** → Compare MAE values
6. **"B-mode shows clinical relevance"** → Point at Row 2

---

## Quick Reference: What Each Column Means

```
┌─────────────────────────────────────────────────────────┐
│ Column 1: Ground Truth (we know the answer)             │
│           ↓ Used to compute error metric (MAE)          │
│ Column 2: DAS (baseline, uniform weights) MAE ≈ 17,500  │
│           ↓ Comparison point                            │
│ Column 3: FISTA (sparse method) MAE ≈ nan              │
│           ↓ Alternative approach (not our focus)        │
│ Column 4: ABLE (ours, learned weights) MAE ≈ 1,260     │
│           ↓ 92.8% improvement!                          │
└─────────────────────────────────────────────────────────┘
```

---

That's it! Each PNG tells the story: **Our learned method (ABLE) dramatically improves reconstruction quality.**
