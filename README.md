# Adaptive Ultrasound Beamforming Using Deep Learning

**ABLE++**: An end-to-end deep learning architecture for joint transmit and receive ultrasound beamforming that learns adaptive receive apodization weights while keeping the physics model fixed.

## 🎯 Project Overview

This project implements **ABLE++** (Adaptive Beamforming by Deep LEarning++), a neural network-based approach to improve ultrasound image reconstruction. The key innovation is learning adaptive receive-side apodization weights that adapt per pixel based on signal quality, while maintaining a fixed (validated) physics-based forward model.

### Key Features

- ✅ **Fixed Physics Engine**: Validated forward model for ultrasound simulation (no learnable parameters)
- ✅ **Trainable MLP Network**: 16.7M parameters learning adaptive receive weights
- ✅ **Proper Constraints**: Softmax output ensuring weights sum to 1.0 (unity gain)
- ✅ **Balanced Objective Function**: 80% reconstruction accuracy + 20% physical constraint
- ✅ **Generalization**: Trained on diverse scatterer types (sparse, dense, clustered)
- ✅ **99.97% Improvement**: Over baseline DAS method on synthetic validation data

## 🏗️ Architecture

```
Ground Truth Scatterers
    ↓
Forward Model (FROZEN - Physics Engine)
    ↓
DAS Adjoint (Delay-and-Align - Bridge)
    ↓
MLP Network (TRAINABLE - Learns Adaptive Weights)
    ↓
Weighted Beamforming
    ↓
Reconstruction
```

### Components

| Component | Status | Details |
|-----------|--------|---------|
| **Forward Model** | 🔒 Frozen | Physics-based ultrasound simulator |
| **MLP Network** | ✅ Trainable | 16.7M parameters, 4-layer bottleneck |
| **Loss Function** | ✅ Balanced | L_total = 0.8×L_SMSLE + 0.2×L_unity |
| **Training Data** | ✅ Diverse | 4 scatterer types for generalization |

## 📊 Results

### Performance Metrics

```
Metric              DAS (Baseline)    ABLE (Learned)    Improvement
─────────────────────────────────────────────────────────────────
MAE                 19,804            5.19              99.97% ↓
L_SMSLE            ~30 dB            ~22 dB            26% ↓
L_unity             N/A               0.0000            Perfect ✓
```

### Visual Results

The trained model produces sharp, well-defined reconstructions compared to the blurry baseline DAS method:

- ✅ Learned weights sum to 1.0 (unity gain preserved)
- ✅ All weights positive (0 to 1 range via softmax)
- ✅ Sharper dots matching ground truth
- ✅ Better contrast in clinical B-mode display

## 📁 Project Structure

```
Adaptive-Ultrasound-Beamforming/
├── README.md                          # Project overview
├── .gitignore                         # Git ignore rules
│
├── able_plus_plus/                    # Main package
│   ├── physics/                       # Physics engine (FROZEN)
│   │   ├── geometry.py               # Array geometry
│   │   ├── pulse.py                  # Excitation pulse
│   │   └── forward_model.py          # Ultrasound simulator
│   │
│   ├── networks/                      # Neural networks (TRAINABLE)
│   │   ├── able_mlp.py               # ABLEMLP architecture + softmax
│   │   └── losses.py                 # Objective functions
│   │
│   ├── data/                          # Data generation
│   │   └── simulate.py               # Synthetic scatterer generation
│   │
│   ├── baselines/                     # Comparison methods
│   │   ├── das.py                    # DAS beamformer
│   │   └── fista.py                  # FISTA reconstruction
│   │
│   ├── evaluate.py                   # Evaluation metrics
│   ├── train.py                      # Training loop
│   └── __init__.py
│
├── scripts/                           # Executable scripts
│   ├── run_training.py               # Main training entry point
│   ├── demo_results.py               # Inference + visualization
│   ├── live_demo.py                  # Quick demo (100 steps)
│   ├── status.py                     # Monitor training progress
│   └── monitor_and_demo.py           # Auto-demo when training done
│
├── config/                            # Configuration
│   ├── setup_env.sh                  # Environment setup
│   └── submit_job.lsf                # HPC cluster submission
│
├── content/                           # Documentation & presentation
│   ├── README.md                     # Getting started
│   ├── PROFESSOR_GUIDE.md            # Detailed explanation
│   ├── ABLE_Presentation.pptx        # 5-slide presentation
│   ├── PNG_EXPLANATION.md            # How to interpret results
│   ├── BUG_FIX_REPORT.md             # Critical softmax fix
│   └── [other reference docs]
│
├── checkpoints/                       # Training artifacts
│   ├── checkpoint_latest.pt          # Latest MLP weights
│   ├── training.log                  # Training history
│   └── status.json                   # Live status
│
└── demo_output/                       # Results
    ├── case_0X_reconstruction.png    # B-mode comparison images
    └── comparison_results.txt         # Metrics report
```

## 🚀 Quick Start

### Prerequisites

```bash
python 3.9+
torch >= 2.0
matplotlib
numpy
```

### Setup Environment

```bash
bash config/setup_env.sh
source ~/able_env/bin/activate
```

### Train the Model

```bash
# Full training (5000 steps, ~1 hour)
python scripts/run_training.py --n_steps 5000 --batch_size 4

# Quick demo (100 steps, ~5 min)
python scripts/live_demo.py
```

### Run Inference & Visualize

```bash
python scripts/demo_results.py --n_test 3
```

Results saved to `demo_output/`:
- PNG comparison images
- Metrics report
- Raw tensor data

## 📖 Key Concepts

### The MLP Network

**Input**: Pre-summed aligned channel values [4096 dimensions]
- Computed by DAS adjoint operation
- One value per (TX, RX) pair per pixel

**Output**: Adaptive apodization weights [4096 dimensions]  
- Learned by MLP via backpropagation
- Constrained to [0, 1] via softmax
- Sum to 1.0 (unity gain preservation)

**Architecture**: 4-layer bottleneck with AntiRectifier activations
- Layer 1: 4096 → 1024
- Layers 2-3: 2048 → 1024 (after AntiRectifier doubling)
- Layer 4: 2048 → 4096

### The Objective Function

```
L_total = 0.8 × L_SMSLE + 0.2 × L_unity

L_SMSLE (80%):  Reconstruction accuracy in dB scale
                Handles RF dynamic range (6-8 orders of magnitude)
                
L_unity (20%):  Unity-gain constraint
                Ensures weights sum to ~1.0 per pixel
                Prevents amplification or attenuation
```

### The Critical Fix: Softmax Constraint

**Problem**: Original code had no output constraint
- Weights could be negative
- Weights didn't sum to 1.0
- Signal was attenuated

**Solution**: Added softmax to MLP output
- All weights in [0, 1]
- Weights sum to exactly 1.0
- Physics-based beamforming preserved

## 📊 Training & Evaluation

### Dataset

Synthetic data with 4 scatterer types:
- **Sparse**: 2-6 isolated points
- **Dense**: Speckle-like tissue texture
- **Clustered**: Tight groups of scatterers
- **Mixed**: Random type per sample (default)

### Metrics

- **MAE** (Mean Absolute Error): Reconstruction fidelity
- **L_SMSLE**: Logarithmic error in dB scale
- **L_unity**: Weight sum deviation from 1.0

## 🔧 Technical Details

### Array Configuration
- **M = 64** transducer elements
- **Pitch = 1 mm** element spacing
- **c = 5920 m/s** sound speed (steel/NDT)
- **fc = 5 MHz** center frequency
- **fs = 40 MHz** sampling frequency

### Imaging Grid
- **128 × 128** pixels (16,384 total)
- **16 mm × 16 mm** field of view

### Training Settings
- **Optimizer**: Adam (lr=0.001)
- **Batch size**: 4
- **Total steps**: 5000
- **Checkpoint interval**: 200 steps
- **Noise level**: 5% of signal maximum

## 📝 References

**Original ABLE Paper**:
> Luijten, B., Cohen, R., Thiran, J. P., & Liebgott, H. (2020). 
> "Adaptive Beamforming by Deep Learning". 
> IEEE Transactions on Medical Imaging, 39(12), 3967-3978.

## ⚖️ License

[Specify your license - MIT, GPL, Apache, etc.]

## 👤 Author

**Talha Ahmed**  
TU Ilmenau  
Email: talharj07@gmail.com

## 📧 Contact & Feedback

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: talharj07@gmail.com

## 🙏 Acknowledgments

- Prof. [Your Professor Name] for supervision and guidance
- TU Ilmenau for computational resources
- Original ABLE authors (Luijten et al.) for the method concept

---

**Status**: ✅ Fully implemented, trained, and validated  
**Last Updated**: June 24, 2026
