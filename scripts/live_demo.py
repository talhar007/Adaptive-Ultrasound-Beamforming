"""Live demo: train a tiny model in seconds and show results.

This script is designed to run RIGHT NOW and demonstrate the ABLE++
pipeline to your professor. It:

  1. Builds a tiny model (M=8 array, 32×32 grid)
  2. Trains the MLP for 100 steps (takes ~30 seconds on GPU)
  3. Generates test cases and compares DAS / MVDR-NS / ABLE
  4. Displays results in the terminal

Run:
    python live_demo.py
"""
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from able_plus_plus import ForwardModel, ABLEMLP, apply_mlp
from able_plus_plus.baselines.das import das_reconstruct
from able_plus_plus.baselines.mvdr import mvdr_reconstruct
from able_plus_plus.data.simulate import random_point_scatterers
from able_plus_plus.evaluate import mae, envelope_db
from able_plus_plus.networks.losses import total_loss


def train_tiny_mlp(model, mlp, n_steps=100, batch_size=2, device='cpu'):
    """Quick training loop — 100 steps takes ~30 sec on A100."""
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    mlp.train()

    losses = []
    print(f"\n  Training MLP for {n_steps} steps...")
    print(f"  ", end='', flush=True)

    for i in range(1, n_steps + 1):
        optimizer.zero_grad()

        # Generate random scatterers
        gt = random_point_scatterers(batch_size, model.nx, model.nz, device=device)

        # Forward model (fixed)
        with torch.no_grad():
            rf = model.simulate(gt)
            rf_noisy = rf + torch.randn_like(rf) * 0.05 * rf.abs().max()

        # DAS adjoint → MLP → loss → backprop
        _, pre_summed = model.das_adjoint(rf_noisy)
        p_recon, weights, _ = apply_mlp(mlp, pre_summed)

        loss, l_img, l_unity = total_loss(p_recon, gt, weights, lam=0.8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

        if i % 20 == 0 or i == n_steps:
            print(f".", end='', flush=True)

    mlp.eval()
    print(f" done. Loss: {losses[0]:.2f} → {losses[-1]:.2f}\n")
    return losses


def demo_compare(model, mlp, device='cpu'):
    """Run inference and compare methods on a few test cases."""
    mlp.eval()

    results = []
    print(f"  Generating test cases and comparing methods...\n")

    for case_idx in range(3):
        gt = random_point_scatterers(1, model.nx, model.nz, device=device)
        n_scatter = (gt > 0).sum().item()

        with torch.no_grad():
            rf = model.simulate(gt)
            rf_noisy = rf + torch.randn_like(rf) * 0.05 * rf.abs().max()

            # Compare methods
            das = das_reconstruct(model, rf_noisy)
            mvdr_ns = mvdr_reconstruct(model, rf_noisy, smoothing=False)

            _, pre_summed = model.das_adjoint(rf_noisy)
            able, weights, _ = apply_mlp(mlp, pre_summed)

            das_mae = mae(das, gt)
            mvdr_ns_mae = mae(mvdr_ns, gt)
            able_mae = mae(able, gt)

        results.append({
            'case': case_idx + 1,
            'n_scatter': n_scatter,
            'das_mae': das_mae,
            'mvdr_ns_mae': mvdr_ns_mae,
            'able_mae': able_mae,
        })

        print(f"    Test {case_idx + 1}: {n_scatter} scatterers")
        print(f"      DAS      MAE: {das_mae:.6f}")
        print(f"      MVDR-NS  MAE: {mvdr_ns_mae:.6f}")
        print(f"      ABLE     MAE: {able_mae:.6f}  ← trained weights")
        print()

    return results


def print_demo_header():
    """Print a nice header."""
    lines = [
        "",
        "╔" + "=" * 68 + "╗",
        "║" + " " * 68 + "║",
        "║" + "  ABLE++ Live Demo — End-to-End Deep Learning Beamforming".center(68) + "║",
        "║" + " " * 68 + "║",
        "║" + "  This script demonstrates the full pipeline:".ljust(68) + "║",
        "║" + "    1. Generate synthetic point scatterers (ground truth)".ljust(68) + "║",
        "║" + "    2. Simulate ultrasound RF data (physics engine, FIXED)".ljust(68) + "║",
        "║" + "    3. Train adaptive apodization weights (ABLE MLP, 100 steps)".ljust(68) + "║",
        "║" + "    4. Compare reconstructions: DAS vs FISTA vs ABLE".ljust(68) + "║",
        "║" + " " * 68 + "║",
        "╚" + "=" * 68 + "╝",
        "",
    ]
    print("\n".join(lines))


def print_demo_results(results):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("  SUMMARY — Reconstruction Error (Lower is Better)")
    print("=" * 70 + "\n")

    print("  Baseline Beamformers (no learning):")
    print("    • DAS (Delay-and-Sum):  uniform weights, simple but effective")
    print("    • MVDR-NS:              adaptive, diagonal-loading only, no spatial smoothing\n")
    print("  Learned Beamformer:")
    print("    • ABLE:                 neural adaptive apodization (trained 100 steps)\n")

    avg_das = sum(r['das_mae'] for r in results) / len(results)
    avg_mvdr_ns = sum(r['mvdr_ns_mae'] for r in results) / len(results)
    avg_able = sum(r['able_mae'] for r in results) / len(results)

    print(f"  Average MAE (Mean Absolute Error):")
    print(f"    DAS    : {avg_das:.6f}")
    print(f"    MVDR-NS: {avg_mvdr_ns:.6f}")
    print(f"    ABLE   : {avg_able:.6f}")
    print()

    if avg_able < avg_das:
        pct = 100 * (1 - avg_able / avg_das)
        print(f"  ✓ ABLE outperforms DAS by {pct:.1f}%")
    else:
        print(f"  Note: With only 100 training steps, results may be preliminary.")

    print("\n" + "=" * 70)
    print("  Key Points for Your Professor")
    print("=" * 70 + "\n")
    print("  1. Fixed Physics Engine:")
    print("     - Forward model is NOT trained — it's a frozen physics simulator")
    print("     - Only the MLP weights are learned")
    print("     - This ensures the learned weights are *adaptive*, not overfitted\n")
    print("  2. Generalization:")
    print("     - Training uses diverse scatterer types (sparse, dense, clustered)")
    print("     - The MLP learns to adapt the apodization weights per-pixel")
    print("     - Works on synthetic data with known ground truth\n")
    print("  3. Pipeline:")
    print("     - Scatterers → [Physics] → RF Data → [DAS Adjoint] → Pre-Summed")
    print("     - Pre-Summed → [MLP] → Weights → Weighted Sum → Image")
    print("     - Loss compares reconstruction vs ground truth (SMSLE + unity penalty)\n")
    print("  4. Next Steps:")
    print("     - Train on larger datasets (5000+ steps)")
    print("     - Evaluate on real ultrasound data")
    print("     - Measure resolution (FWHM), contrast (CNR), reconstruction fidelity (MAE)\n")
    print("=" * 70 + "\n")


def main():
    print_demo_header()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // 2**30} GB\n")

    # Tiny model for speed
    M, nx, nz = 8, 32, 32
    print(f"Building tiny model: M={M}, grid={nx}×{nz}")

    model = ForwardModel(M=M, nx=nx, nz=nz, device=device).to(device)
    mlp = ABLEMLP(N=M * M).to(device)

    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"  Forward Model: fixed (physics engine, no learning)")
    print(f"  ABLEMLP: {n_params:,} trainable parameters\n")

    # Quick training
    print("STEP 1: Train the MLP (100 steps)")
    print("-" * 70)
    losses = train_tiny_mlp(model, mlp, n_steps=100, batch_size=2, device=device)

    # Inference & comparison
    print("STEP 2: Evaluate on Test Cases")
    print("-" * 70)
    results = demo_compare(model, mlp, device=device)

    # Summary
    print_demo_results(results)

    print(f"Demo completed at {datetime.now().strftime('%H:%M:%S')}\n")


if __name__ == '__main__':
    main()
