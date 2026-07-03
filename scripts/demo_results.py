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
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from able_plus_plus import ForwardModel, ABLEMLP, apply_mlp
from able_plus_plus.baselines.das import das_reconstruct
from able_plus_plus.baselines.fista import fista_reconstruct
from able_plus_plus.data.simulate import random_scatterer_batch
from able_plus_plus.evaluate import mae
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


def to_common_scale(flat_img, nz, nx, eps=1e-8):
    """THE one shared normalization for GT / DAS / FISTA / ABLE alike:
    flat [P] reconstruction -> [nz, nx] magnitude, peak-normalized to [0, 1].

    - abs() handles bipolar RF-domain outputs (DAS, ABLE) and is a no-op
      for non-negative sparse outputs (GT, FISTA).
    - Peak normalization is correct for both sparse maps (>99% zero pixels,
      where percentile-based scales collapse to 0) and dense maps.
    - This matches the per-sample peak normalization inside smsle_loss, so
      training loss, MAE and the B-mode display all share one scale.
    """
    mag = np.abs(np.asarray(flat_img).reshape(nz, nx))
    peak = max(float(mag.max()), eps)
    return np.clip(mag / peak, 0.0, 1.0)


def to_bmode(norm_img):
    """Shared dB compression for display, applied to peak-normalized maps.

    norm^1.5 compression: threshold = 10% of peak (norm > 0.1 visible at
    vmin=-30). Softer than norm^2 (17.8% threshold, hid weak scatterers)
    and harder than norm (3.16% threshold, too much clutter).
    """
    return 20 * np.log10(norm_img ** 1.5 + 1e-12)


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

    # Cycle deterministically through sparse → dense → clustered so every run
    # includes all three field types. 'mixed' would pick randomly and can
    # accidentally draw all-sparse batches, biasing MAE toward FISTA (whose L1
    # prior is optimal for sparse but wrong for dense/clustered).
    EVAL_TYPES = ['sparse', 'dense', 'clustered']

    for case_idx in range(n_test_cases):
        stype = EVAL_TYPES[case_idx % len(EVAL_TYPES)]
        print(f"\n--- Test case {case_idx + 1}/{n_test_cases} ({stype}) ---")

        gt = random_scatterer_batch(1, model.nx, model.nz,
                                    scatterer_type=stype, device=device)
        n_scatterers = int(torch.sum(gt > 0).item())
        print(f"  GT: {n_scatterers} scatterers")

        # Simulate RF data
        with torch.no_grad():
            rf = model.simulate(gt)
            rf_noisy = rf + torch.randn_like(rf) * 0.05 * rf.abs().max()
        print(f"  RF data: {tuple(rf_noisy.shape)}")

        # DAS baseline
        das = das_reconstruct(model, rf_noisy)

        # FISTA: step size and lambda are both adaptive (see fista.py).
        fista = fista_reconstruct(model, rf_noisy)
        fista_valid = not torch.isnan(fista).any()

        # ABLE (trained MLP)
        with torch.no_grad():
            _, pre_summed = model.das_adjoint(rf_noisy)
            able, weights, _ = apply_mlp(mlp, pre_summed)

        # One shared pipeline (magnitude + peak-normalized [0,1] scaling)
        # for GT/DAS/FISTA/ABLE alike, so MAE and the B-mode images are
        # always computed on the same common scale.
        gt_norm = to_common_scale(gt[0].cpu().numpy(), model.nz, model.nx)
        das_norm = to_common_scale(das[0].cpu().numpy(), model.nz, model.nx)
        able_norm = to_common_scale(able[0].cpu().numpy(), model.nz, model.nx)
        if fista_valid:
            fista_norm = to_common_scale(fista[0].cpu().numpy(), model.nz, model.nx)
        else:
            fista_norm = np.full((model.nz, model.nx), np.nan)

        das_mae = mae(torch.from_numpy(das_norm), torch.from_numpy(gt_norm))
        fista_mae = mae(torch.from_numpy(fista_norm), torch.from_numpy(gt_norm)) if fista_valid else float('nan')
        able_mae = mae(torch.from_numpy(able_norm), torch.from_numpy(gt_norm))

        print(f"  DAS MAE: {das_mae:.4f}")
        print(f"  FISTA MAE: {fista_mae:.4f}" if fista_valid else "  FISTA MAE: nan")
        print(f"  ABLE MAE: {able_mae:.4f}")

        # Compute loss for ABLE
        with torch.no_grad():
            l_total, l_img, l_unity = total_loss(able, gt, weights, lam=0.8)

        results.append({
            'case':        case_idx + 1,
            'n_scatterers': n_scatterers,
            'gt':          gt_norm,
            'das':         das_norm,
            'fista':       fista_norm,
            'able':        able_norm,
            'gt_bmode':    to_bmode(gt_norm),
            'das_bmode':   to_bmode(das_norm),
            'fista_bmode': to_bmode(fista_norm),
            'able_bmode':  to_bmode(able_norm),
            'das_mae':     das_mae,
            'fista_mae':   fista_mae,
            'able_mae':    able_mae,
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
        lines.append(f"  Ground Truth: {res['n_scatterers']} scatterers")
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
    except ImportError:
        print("matplotlib not installed — skipping visualization")
        return

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    for res in results:
        case = res['case']
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Test Case {case}: B-mode Reconstruction (dB, common scale)', fontsize=14)

        # B-mode only (clinical display)
        axes[0].imshow(res['gt_bmode'], cmap='gray', vmin=-30, vmax=0)
        axes[0].set_title("Ground Truth\n(B-mode)")
        axes[0].axis('off')

        axes[1].imshow(res['das_bmode'], cmap='gray', vmin=-30, vmax=0)
        axes[1].set_title(f"DAS\nMAE: {res['das_mae']:.4f}")
        axes[1].axis('off')

        axes[2].imshow(res['fista_bmode'], cmap='gray', vmin=-30, vmax=0)
        axes[2].set_title(f"FISTA\nMAE: {res['fista_mae']:.4f}")
        axes[2].axis('off')

        axes[3].imshow(res['able_bmode'], cmap='gray', vmin=-30, vmax=0)
        axes[3].set_title(f"ABLE\nMAE: {res['able_mae']:.4f}")
        axes[3].axis('off')

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


if __name__ == '__main__':
    main()
