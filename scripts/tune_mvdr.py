"""Tune MVDR / MVDR-NS parameters against the DAS baseline.

The professor flagged the shipped `mvdr_reconstruct` defaults as "certainly
wrong" -- MVDR/MVDR-NS underperform even plain DAS. The root-cause
hypothesis (see the implementation plan): MVDR-NS (`smoothing=False`) builds
a RANK-1 single-snapshot covariance matrix, and the current default
`loading='eigen'` gives near-zero regularization for a matrix with M-1
near-zero eigenvalues by construction -- a near-guaranteed numerical
explosion. The paper's own recipe (`loading='trace', diag_load=0.1`) should
fix this outright. This script measures that hypothesis directly, then
sweeps `L` (spatial-smoothing subaperture length) for the smoothed `mvdr`
variant, and reports MAE / PSNR / CNR for every config against DAS.

Usage:
    python scripts/tune_mvdr.py
    python scripts/tune_mvdr.py --n_cases 12 --M 64
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from able_plus_plus import ForwardModel
from able_plus_plus.baselines.das import das_reconstruct
from able_plus_plus.baselines.mvdr import mvdr_reconstruct
from able_plus_plus.data.simulate import random_scatterer_batch
from able_plus_plus.evaluate import mae, psnr, scatterer_cnr

sys.path.insert(0, str(Path(__file__).parent))
from demo_results import to_common_scale, to_bmode  # noqa: E402

NOISE_LEVEL = 0.05
EVAL_TYPES = ['sparse', 'dense', 'clustered']


def run_config(model, cases, label, **mvdr_kwargs):
    """Reconstruct every case with mvdr_reconstruct(**mvdr_kwargs) and
    return averaged (mae, psnr, cnr) against ground truth, plus whether any
    case produced a NaN/Inf (a real risk with under-loaded inversions)."""
    maes, psnrs, cnrs = [], [], []
    n_bad = 0
    for gt, rf_noisy in cases:
        img = mvdr_reconstruct(model, rf_noisy, **mvdr_kwargs)
        if not torch.isfinite(img).all():
            n_bad += 1
            continue
        gt_norm = to_common_scale(gt[0].cpu().numpy(), model.nz, model.nx)
        norm = to_common_scale(img[0].cpu().numpy(), model.nz, model.nx)
        bmode = to_bmode(norm)
        maes.append(mae(torch.from_numpy(norm), torch.from_numpy(gt_norm)))
        psnrs.append(psnr(norm, gt_norm))
        cnrs.append(scatterer_cnr(bmode, gt_norm))
    n = max(len(maes), 1)
    return {
        'label': label,
        'mae':  float(np.mean(maes)) if maes else float('nan'),
        'psnr': float(np.nanmean(psnrs)) if psnrs else float('nan'),
        'cnr':  float(np.nanmean(cnrs)) if cnrs else float('nan'),
        'n_bad': n_bad,
        'n': n,
    }


def print_table(rows):
    header = f"  {'config':42s}  {'MAE':>9s}  {'PSNR(dB)':>9s}  {'CNR(dB)':>9s}  {'bad':>4s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {r['label']:42s}  {r['mae']:9.6f}  {r['psnr']:9.2f}  "
              f"{r['cnr']:9.2f}  {r['n_bad']:4d}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n_cases', type=int, default=9,
                   help='test cases (cycled sparse/dense/clustered)')
    p.add_argument('--M', type=int, default=64)
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--nz', type=int, default=128)
    p.add_argument('--max_points', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    model = ForwardModel(M=args.M, nx=args.nx, nz=args.nz, device=device).to(device)

    # Build a fixed batch of test cases up front so every config is measured
    # on identical data.
    cases = []
    for i in range(args.n_cases):
        stype = EVAL_TYPES[i % len(EVAL_TYPES)]
        gt = random_scatterer_batch(1, model.nx, model.nz, scatterer_type=stype,
                                    device=device, max_points=args.max_points)
        with torch.no_grad():
            rf = model.simulate(gt)
            rf_noisy = rf + torch.randn_like(rf) * NOISE_LEVEL * rf.abs().max()
        cases.append((gt, rf_noisy))
    print(f"Built {len(cases)} test cases (M={args.M}, grid={args.nx}x{args.nz})\n")

    das_maes, das_psnrs, das_cnrs = [], [], []
    for gt, rf_noisy in cases:
        img = das_reconstruct(model, rf_noisy)
        gt_norm = to_common_scale(gt[0].cpu().numpy(), model.nz, model.nx)
        norm = to_common_scale(img[0].cpu().numpy(), model.nz, model.nx)
        bmode = to_bmode(norm)
        das_maes.append(mae(torch.from_numpy(norm), torch.from_numpy(gt_norm)))
        das_psnrs.append(psnr(norm, gt_norm))
        das_cnrs.append(scatterer_cnr(bmode, gt_norm))
    rows = [{'label': 'DAS (reference)', 'mae': float(np.mean(das_maes)),
            'psnr': float(np.mean(das_psnrs)), 'cnr': float(np.nanmean(das_cnrs)),
            'n_bad': 0, 'n': len(cases)}]

    print("=" * 80)
    print("  Step 1: primary hypothesis -- MVDR-NS (smoothing=False), current")
    print("  default (loading='eigen', kappa=1e3) vs. the paper's own recipe")
    print("  (loading='trace', diag_load=0.1). Rank-1 covariance argument")
    print("  predicts the current default is the bug.")
    print("=" * 80)
    rows.append(run_config(model, cases, "mvdr_ns  current default (eigen, kappa=1e3)",
                           smoothing=False, loading='eigen', kappa=1e3))
    rows.append(run_config(model, cases, "mvdr_ns  paper recipe (trace, diag_load=0.1)",
                           smoothing=False, loading='trace', diag_load=0.1))
    for kappa in (1e2, 1e4):
        rows.append(run_config(model, cases, f"mvdr_ns  eigen, kappa={kappa:g}",
                               smoothing=False, loading='eigen', kappa=kappa))
    for dl in (0.03, 0.3):
        rows.append(run_config(model, cases, f"mvdr_ns  trace, diag_load={dl:g}",
                               smoothing=False, loading='trace', diag_load=dl))

    print("\n" + "=" * 80)
    print("  Step 2: smoothed MVDR -- current default (L=M//2, eigen) vs.")
    print("  paper-ratio L (~25% of M) x tuned loading from Step 1.")
    print("=" * 80)
    M = args.M
    rows.append(run_config(model, cases, f"mvdr     current default (L={M//2}, eigen, kappa=1e3)",
                           smoothing=True, L=M // 2, loading='eigen', kappa=1e3))
    for L in (M // 4, M // 2):
        for loading, kw in (('trace', dict(diag_load=0.1)), ('eigen', dict(kappa=1e3))):
            rows.append(run_config(model, cases, f"mvdr     L={L}, loading={loading}",
                                   smoothing=True, L=L, loading=loading, **kw))

    print()
    print_table(rows)

    best = min((r for r in rows if r['label'] != 'DAS (reference)' and np.isfinite(r['mae'])),
              key=lambda r: r['mae'], default=None)
    print()
    if best is not None and best['mae'] < rows[0]['mae']:
        print(f"Best config beats DAS on MAE: {best['label']}")
    else:
        print("No MVDR config clearly beat DAS on MAE in this sweep -- per the "
              "professor's explicit fallback, drop the smoothed 'mvdr' column "
              "and keep only a correctly-loaded MVDR-NS if it beats DAS, else "
              "flag for further investigation before changing any defaults.")


if __name__ == '__main__':
    main()
