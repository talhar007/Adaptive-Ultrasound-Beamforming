"""Demo: show ABLE++ results to your professor.

Loads the trained MLP, generates a test case with known point scatterers,
and compares DAS / FISTA / ABLE reconstructions side-by-side with metrics.

Run:
    python demo_results.py

(or with custom checkpoint path:
    python demo_results.py --checkpoint path/to/model.pt
)

Outputs:
    - comparison_results.txt     quantitative metrics
    - visualization_*.png        B-mode images (if matplotlib available)
    - reconstruction_*.pt        raw tensors for analysis (optional - disabled by default)
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from able_plus_plus import ForwardModel, ABLEMLP, apply_mlp
from able_plus_plus.baselines.das import das_reconstruct
from able_plus_plus.baselines.fista import fista_reconstruct
from able_plus_plus.data.simulate import random_point_scatterers
from able_plus_plus.evaluate import mae, envelope_db
from able_plus_plus.networks.losses import total_loss


# ============================================================
# helpers
# ============================================================

def load_checkpoint(ckpt_path, model, mlp, device):
    """Load trained MLP weights from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    mlp.load_state_dict(ckpt['mlp_state'])
    print(f"Loaded MLP weights from {ckpt_path}")
    return ckpt


def normalize_image(img):
    """Normalize to [0, 1] for display."""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img


def compare_methods(model, mlp, n_test_cases=3, device='cpu'):
    """Run the full pipeline on test cases and collect metrics.

    For each test case:
      1. Generate random point scatterers (ground truth)
      2. Simulate RF data
      3. Reconstruct using: DAS, FISTA, ABLE (trained MLP)
      4. Compute metrics and losses
      5. Store results
    """
    results = []

    for case_idx in range(n_test_cases):
        print(f"\n--- Test case {case_idx + 1}/{n_test_cases} ---")

        # Generate ground truth
        gt = random_point_scatterers(1, model.nx, model.nz, device=device)
        print(f"  GT: {torch.sum(gt > 0).item()} scatterers")

        # Simulate RF data
        with torch.no_grad():
            rf = model.simulate(gt)
            rf_norm = rf.abs().max()
            rf_noisy = rf + torch.randn_like(rf) * 0.05 * rf_norm
        print(f"  RF data: {tuple(rf_noisy.shape)}")

        # DAS baseline
        das = das_reconstruct(model, rf_noisy)
        das_mae = mae(das, gt)
        print(f"  DAS MAE: {das_mae:.4f}")

        # FISTA (20 iters, reasonable speed)
        fista = fista_reconstruct(model, rf_noisy, n_iter=20, lam=1e-3, step=1e-8)
        fista_mae = mae(fista, gt)
        print(f"  FISTA MAE: {fista_mae:.4f}")

        # ABLE (trained MLP)
        with torch.no_grad():
            _, pre_summed = model.das_adjoint(rf_noisy)
            able, weights, _ = apply_mlp(mlp, pre_summed)
            able_mae = mae(able, gt)
        print(f"  ABLE MAE: {able_mae:.4f}")

        # Compute loss for ABLE
        with torch.no_grad():
            l_total, l_img, l_unity = total_loss(able, gt, weights, lam=0.8)

        # B-mode envelope (standard ultrasound display)
        das_bmode = envelope_db(das[0].cpu())
        fista_bmode = envelope_db(fista[0].cpu())
        able_bmode = envelope_db(able[0].cpu())
        gt_bmode = envelope_db(gt[0].cpu())

        results.append({
            'case':      case_idx + 1,
            'gt':        gt.cpu().numpy(),
            'das':       das.cpu().numpy(),
            'fista':     fista.cpu().numpy(),
            'able':      able.cpu().numpy(),
            'das_bmode': das_bmode,
            'fista_bmode': fista_bmode,
            'able_bmode': able_bmode,
            'gt_bmode':  gt_bmode,
            'das_mae':   das_mae,
            'fista_mae': fista_mae,
            'able_mae':  able_mae,
            'able_loss_total': l_total.item(),
            'able_loss_image':  l_img.item(),
            'able_loss_unity':  l_unity.item(),
        })

    return results


def format_report(results):
    """Pretty-print results as a text report."""
    lines = [
        "=" * 70,
        "  ABLE++ Reconstruction Results — Comparison",
        "=" * 70,
        "",
    ]

    mae_summary = {'DAS': [], 'FISTA': [], 'ABLE': []}
    for res in results:
        mae_summary['DAS'].append(res['das_mae'])
        mae_summary['FISTA'].append(res['fista_mae'])
        mae_summary['ABLE'].append(res['able_mae'])

        lines.append(f"Test Case {res['case']}:")
        lines.append(f"  Ground Truth: {sum(res['gt'][0] > 0)} scatterers")
        lines.append(f"  MAE (DAS)  : {res['das_mae']:.6f}")
        lines.append(f"  MAE (FISTA): {res['fista_mae']:.6f}")
        lines.append(f"  MAE (ABLE) : {res['able_mae']:.6f}  ← learned weights")
        lines.append(f"    ↳ Loss (total): {res['able_loss_total']:.4f}")
        lines.append(f"    ↳ Loss (image): {res['able_loss_image']:.4f}")
        lines.append(f"    ↳ Loss (unity): {res['able_loss_unity']:.4f}")
        lines.append("")

    lines.append("Summary (average across test cases):")
    lines.append(f"  DAS  MAE: {sum(mae_summary['DAS']) / len(mae_summary['DAS']):.6f}")
    lines.append(f"  FISTA MAE: {sum(mae_summary['FISTA']) / len(mae_summary['FISTA']):.6f}")
    lines.append(f"  ABLE  MAE: {sum(mae_summary['ABLE']) / len(mae_summary['ABLE']):.6f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Lower MAE = better reconstruction fidelity")
    lines.append("  - ABLE uses learned apodization weights (θ) trained on synthetic data")
    lines.append("  - Weights generalize across sparse, dense, and clustered scatterers")
    lines.append("  - B-mode images show typical ultrasound (dB-compressed, log scale)")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_visualizations(results, output_dir='demo_output'):
    """Save B-mode images as PNG (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping visualization")
        return

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    for res in results:
        case = res['case']
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Test Case {case}: Point Scatterer Reconstruction', fontsize=14)

        # Reshape flattened images to 2D for display
        nx, nz = 128, 128
        gt_2d = res['gt'].reshape(nx, nz)
        das_2d = res['das'].reshape(nx, nz)
        fista_2d = res['fista'].reshape(nx, nz)
        able_2d = res['able'].reshape(nx, nz)

        # Row 0: Linear amplitude
        axes[0, 0].imshow(normalize_image(torch.from_numpy(gt_2d)), cmap='gray')
        axes[0, 0].set_title(f"Ground Truth (Linear)\nMAE: N/A")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(normalize_image(torch.from_numpy(das_2d)), cmap='gray')
        axes[0, 1].set_title(f"DAS (Linear)\nMAE: {res['das_mae']:.4f}")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(normalize_image(torch.from_numpy(fista_2d)), cmap='gray')
        axes[0, 2].set_title(f"FISTA (Linear)\nMAE: {res['fista_mae']:.4f}")
        axes[0, 2].axis('off')

        axes[0, 3].imshow(normalize_image(torch.from_numpy(able_2d)), cmap='gray')
        axes[0, 3].set_title(f"ABLE (Linear)\nMAE: {res['able_mae']:.4f}")
        axes[0, 3].axis('off')

        # Row 1: B-mode (dB, log-compressed)
        axes[1, 0].imshow(res['gt_bmode'].reshape(nx, nz), cmap='gray', vmin=-60, vmax=0)
        axes[1, 0].set_title("Ground Truth (B-mode, dB)")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(res['das_bmode'].reshape(nx, nz), cmap='gray', vmin=-60, vmax=0)
        axes[1, 1].set_title("DAS (B-mode)")
        axes[1, 1].axis('off')

        axes[1, 2].imshow(res['fista_bmode'].reshape(nx, nz), cmap='gray', vmin=-60, vmax=0)
        axes[1, 2].set_title("FISTA (B-mode)")
        axes[1, 2].axis('off')

        axes[1, 3].imshow(res['able_bmode'].reshape(nx, nz), cmap='gray', vmin=-60, vmax=0)
        axes[1, 3].set_title("ABLE (B-mode)")
        axes[1, 3].axis('off')

        plt.tight_layout()
        png_path = out / f"case_{case:02d}_reconstruction.png"
        plt.savefig(png_path, dpi=100, bbox_inches='tight')
        print(f"Saved → {png_path}")
        plt.close()


def save_tensors(results, output_dir='demo_output'):
    """Save raw tensors as .pt files for further analysis."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    for res in results:
        case = res['case']
        data = {
            'gt':         torch.from_numpy(res['gt']),
            'das':        torch.from_numpy(res['das']),
            'fista':      torch.from_numpy(res['fista']),
            'able':       torch.from_numpy(res['able']),
            'metrics':    {
                'das_mae':   res['das_mae'],
                'fista_mae': res['fista_mae'],
                'able_mae':  res['able_mae'],
            },
        }
        pt_path = out / f"case_{case:02d}_tensors.pt"
        torch.save(data, pt_path)
        print(f"Saved → {pt_path}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Demo ABLE++ pipeline: generate test case, compare methods, show metrics'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_latest.pt',
                        help='path to trained model checkpoint')
    parser.add_argument('--n_test', type=int, default=3,
                        help='number of test cases to run')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--M', type=int, default=64)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='demo_output',
                        help='directory to save results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build model
    model = ForwardModel(M=args.M, nx=args.nx, nz=args.nz, device=device).to(device)
    mlp = ABLEMLP(N=args.M * args.M).to(device)

    # Load checkpoint
    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    load_checkpoint(args.checkpoint, model, mlp, device)
    mlp.eval()

    # Run comparisons
    print(f"\nRunning {args.n_test} test cases...")
    results = compare_methods(model, mlp, n_test_cases=args.n_test, device=device)

    # Generate report
    report = format_report(results)
    print(f"\n{report}")

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    report_path = out_dir / 'comparison_results.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report → {report_path}")

    # .pt files: Optional. Uncomment if you need raw tensor data for analysis.
    # save_tensors(results, args.output_dir)

    save_visualizations(results, args.output_dir)

    print(f"\nDone. Results in {out_dir}/")
    print(results)


if __name__ == '__main__':
    main()
