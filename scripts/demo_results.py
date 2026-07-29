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
from able_plus_plus.networks.able_mlp import pixel_positions
from able_plus_plus.networks.tx_params import TxParams
from able_plus_plus.baselines.das import das_reconstruct
from able_plus_plus.baselines.fista import fista_reconstruct
from able_plus_plus.baselines.mvdr import mvdr_reconstruct
from able_plus_plus.data.simulate import random_scatterer_batch
from able_plus_plus.evaluate import mae, psnr, smsle, scatterer_cnr
from able_plus_plus.networks.losses import total_loss

METHOD_KEYS  = ['das', 'fista', 'mvdr', 'mvdr_ns', 'able']
METHOD_NAMES = {'das': 'DAS', 'fista': 'FISTA', 'mvdr': 'MVDR',
                'mvdr_ns': 'MVDR-NS', 'able': 'ABLE', 'able_pp': 'ABLE++',
                'able_pps': 'ABLE++S'}

# Imaging grid extent, matching ForwardModel's defaults (x_lim=32e-3,
# z_start=10e-3, z_end=74e-3, pitch=1e-3 m) — demo_results.py always
# constructs ForwardModel without overriding these, so the display axes
# below are accurate for every plot this script produces.
X_LIM_MM   = 32.0
Z_START_MM = 10.0
Z_END_MM   = 74.0
PITCH_MM   = 1.0

# Additive noise level used everywhere in this script: rf += N(0,1) *
# NOISE_LEVEL * rf.abs().max() (5% of the peak clean-signal amplitude).
NOISE_LEVEL = 0.05


def _imaging_info_lines(model, max_points, forward_name, elem_dropout=0.0):
    """Two-line summary of the imaging grid, array/physics parameters, and
    acquisition settings shared by every panel/case in a run — so every
    output (figure captions and the text report) states exactly what was
    reconstructed and under what conditions, not just the per-method
    metrics.
    """
    nx, nz = model.nx, model.nz
    dx_mm = 2 * X_LIM_MM / (nx - 1)
    dz_mm = (Z_END_MM - Z_START_MM) / (nz - 1)
    line1 = (f"Image grid: {nx}x{nz} px  |  Lateral ±{X_LIM_MM:.0f} mm "
            f"(Δx={dx_mm:.2f} mm)  |  Depth {Z_START_MM:.0f}-{Z_END_MM:.0f} mm "
            f"(Δz={dz_mm:.2f} mm)  |  Array: M={model.M}, pitch={PITCH_MM:.1f} mm")
    line2 = (f"Physics: c={model.c:.0f} m/s, fc={model.fc/1e6:.1f} MHz, "
            f"fs={model.fs/1e6:.0f} MHz  |  Noise {NOISE_LEVEL*100:.0f}% of peak  |  "
            f"Max {max_points} scatterers/case  |  Forward: {forward_name}")
    if elem_dropout > 0:
        line2 += f"  |  Damaged aperture: {elem_dropout*100:.0f}% elements dead"
    return line1, line2


# ============================================================
# helpers
# ============================================================

def load_network(ckpt_path, N, device, nx, nz):
    """Build the ABLEMLP recorded in a checkpoint (with positional encoding
    if it was trained with --pos_enc) and load its weights.

    returns (mlp, ckpt, pos) — pos is the [P, 2] positional-feature tensor
    for apply_mlp, or None for paper-faithful (position-agnostic) models.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    pos_enc = ckpt.get('config', {}).get('pos_enc', False)
    mlp = ABLEMLP(N=N, pos_dim=2 if pos_enc else 0).to(device)
    mlp.load_state_dict(ckpt['mlp_state'])
    pos = pixel_positions(nx, nz, device) if pos_enc else None
    print(f"Loaded ABLEMLP weights from {ckpt_path}"
          + (" (positional encoding)" if pos_enc else ""))
    return mlp, ckpt, pos


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
    """Shared dB compression for display: STANDARD B-mode log compression,
    20*log10(norm), over the -60 dB display range.

    Earlier revisions compressed norm^1.5, which pushes everything below
    ~1% of peak to black — an aggressive visibility threshold that hid
    low-level content: residual noise, weak echoes, and the visible
    differences between ABLE and ABLE++. The standard compression keeps
    everything down to 0.1% of peak visible: more noise on screen, but no
    information suppressed within the display range.
    """
    return 20 * np.log10(norm_img + 1e-12)


def _reconstruct_tx_variant(variant, gt, sim, model, elem_mask):
    """Shared reconstruction path for any joint-TX+RX checkpoint (ABLE++,
    ABLE++S, ...): acquire with its learned w_tx/tau_tx via simulate_tx,
    then the ordinary frozen das_adjoint and its own receive MLP.
    variant: (mlp, tx, pos) tuple, as loaded by load_network + TxParams."""
    mlp_v, tx_v, pos_v = variant
    with torch.no_grad():
        w_tx, tau_tx = tx_v()
        if elem_mask is not None:
            w_tx = w_tx * elem_mask
        rf_v = sim.simulate_tx(gt, w_tx=w_tx, tau_tx=tau_tx)
        if elem_mask is not None:
            rf_v = rf_v * elem_mask.view(1, -1, 1)
        rf_v = rf_v + torch.randn_like(rf_v) * NOISE_LEVEL * rf_v.abs().max()
        _, pre_v = model.das_adjoint(rf_v)
        out, _, _ = apply_mlp(mlp_v, pre_v, pos=pos_v)
    return out


def compare_methods(model, mlp, n_test_cases=3, device='cpu', max_points=20,
                    pp=None, pps=None, sim_model=None, pos=None, elem_mask=None):
    """Run the full pipeline on test cases and collect metrics.

    For each test case:
      1. Generate random point scatterers (ground truth, <= max_points so
         results stay visually countable)
      2. Simulate RF data
      3. Reconstruct using: DAS, FISTA, MVDR, ABLE (trained network)
      4. Compute MAE / PSNR / SMSLE / CNR per method, plus ABLE's loss
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
                                    scatterer_type=stype, device=device,
                                    max_points=max_points)
        n_scatterers = int(torch.sum(gt > 0).item())
        print(f"  GT: {n_scatterers} scatterers")

        # Simulate RF data. sim_model may differ from the reconstruction
        # model (physics-mismatch study: e.g. deepwave-generated RF
        # reconstructed with the analytic operators). elem_mask simulates a
        # damaged aperture — dead elements neither transmit (w_tx=0) nor
        # receive (rf channel zeroed) — identically for every method.
        sim = sim_model or model
        with torch.no_grad():
            if elem_mask is None:
                rf = sim.simulate(gt)
            else:
                rf = sim.simulate_tx(gt, w_tx=elem_mask) * elem_mask.view(1, -1, 1)
            rf_noisy = rf + torch.randn_like(rf) * NOISE_LEVEL * rf.abs().max()
        print(f"  RF data: {tuple(rf_noisy.shape)}")

        # Baselines: uniform sum, sparse iterative, classical adaptive
        # (with spatial smoothing / full-aperture single-snapshot).
        recon = {
            'das':     das_reconstruct(model, rf_noisy),
            'fista':   fista_reconstruct(model, rf_noisy),
            'mvdr':    mvdr_reconstruct(model, rf_noisy),
            'mvdr_ns': mvdr_reconstruct(model, rf_noisy, smoothing=False),
        }

        # ABLE (trained network, receive-only — the paper's method)
        with torch.no_grad():
            _, pre_summed = model.das_adjoint(rf_noisy)
            able, weights, _ = apply_mlp(mlp, pre_summed, pos=pos)
        recon['able'] = able

        # ABLE++ (joint TX+RX learning): the ACQUISITION itself uses the
        # learned transmit apodization + firing delays — simulate_tx runs
        # on the transmit side — then the ordinary frozen das_adjoint and
        # its own receive MLP reconstruct. Same GT, same noise level.
        # Under elem_mask, the learned firing is masked by the same dead
        # elements as everyone else. ABLE++S (pps) is the SAME mechanism
        # with a smoothness-regularized TxParams checkpoint — kept as a
        # separate column so the un-regularized ABLE++ stays available
        # for comparison, not overwritten.
        if pp is not None:
            recon['able_pp'] = _reconstruct_tx_variant(pp, gt, sim, model, elem_mask)
        if pps is not None:
            recon['able_pps'] = _reconstruct_tx_variant(pps, gt, sim, model, elem_mask)

        # One shared pipeline (magnitude + peak-normalized [0,1] scaling)
        # for GT and every method alike, so all metrics and the B-mode
        # images are computed on the same common scale.
        gt_norm = to_common_scale(gt[0].cpu().numpy(), model.nz, model.nx)
        res = {
            'case':         case_idx + 1,
            'type':         stype,
            'n_scatterers': n_scatterers,
            'gt':           gt_norm,
            'gt_bmode':     to_bmode(gt_norm),
            'methods':      {},
        }

        for key in METHOD_KEYS:
            img = recon[key]
            valid = not torch.isnan(img).any()
            if valid:
                norm  = to_common_scale(img[0].cpu().numpy(), model.nz, model.nx)
                bmode = to_bmode(norm)
                m = {
                    'norm':  norm,
                    'bmode': bmode,
                    'mae':   mae(torch.from_numpy(norm), torch.from_numpy(gt_norm)),
                    'psnr':  psnr(norm, gt_norm),
                    'smsle': smsle(norm, gt_norm),
                    'cnr':   scatterer_cnr(bmode, gt_norm),
                }
            else:
                nan_img = np.full((model.nz, model.nx), np.nan)
                m = {'norm': nan_img, 'bmode': nan_img,
                     'mae': float('nan'), 'psnr': float('nan'),
                     'smsle': float('nan'), 'cnr': float('nan')}
            res['methods'][key] = m
            print(f"  {METHOD_NAMES[key]:5s}  MAE {m['mae']:.4f}   "
                  f"PSNR {m['psnr']:6.2f} dB   SMSLE {m['smsle']:.4f}   "
                  f"CNR {m['cnr']:6.2f} dB")

        # Training-loss values for ABLE (reporting only)
        with torch.no_grad():
            l_total, l_img, l_unity = total_loss(able, gt, weights, lam=0.8)
        res['able_loss_total'] = l_total.item()
        res['able_loss_image'] = l_img.item()
        res['able_loss_unity'] = l_unity.item()

        results.append(res)

    return results


def format_report(results, model, max_points, forward_name, elem_dropout=0.0):
    """Pretty-print results as a text report."""
    info1, info2 = _imaging_info_lines(model, max_points, forward_name, elem_dropout)
    lines = [
        "=" * 70,
        "  ABLE++ Reconstruction Results — Comparison",
        "=" * 70,
        info1,
        info2,
        "=" * 70,
        "",
    ]

    header = f"  {'method':6s}  {'MAE':>9s}  {'PSNR(dB)':>9s}  {'SMSLE':>9s}  {'CNR(dB)':>9s}"
    summary = {k: {'mae': [], 'psnr': [], 'smsle': [], 'cnr': []} for k in METHOD_KEYS}

    for res in results:
        lines.append(f"Test Case {res['case']} ({res['type']}):")
        lines.append(f"  Ground Truth: {res['n_scatterers']} scatterers")
        lines.append(header)
        for key in METHOD_KEYS:
            m = res['methods'][key]
            mark = '  ← learned weights' if key in ('able', 'able_pp', 'able_pps') else ''
            lines.append(f"  {METHOD_NAMES[key]:6s}  {m['mae']:9.6f}  {m['psnr']:9.2f}"
                         f"  {m['smsle']:9.4f}  {m['cnr']:9.2f}{mark}")
            for metric in summary[key]:
                summary[key][metric].append(m[metric])
        lines.append(f"    ↳ ABLE loss — total: {res['able_loss_total']:.4f}"
                     f" | image: {res['able_loss_image']:.4f}"
                     f" | unity: {res['able_loss_unity']:.4f}")
        lines.append("")

    lines.append("Summary (average across test cases):")
    lines.append(header)
    for key in METHOD_KEYS:
        avg = {metric: float(np.nanmean(vals)) for metric, vals in summary[key].items()}
        lines.append(f"  {METHOD_NAMES[key]:6s}  {avg['mae']:9.6f}  {avg['psnr']:9.2f}"
                     f"  {avg['smsle']:9.4f}  {avg['cnr']:9.2f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - MAE / SMSLE: lower = better fidelity (linear / log domain, Eq. 16)")
    lines.append("  - PSNR / CNR : higher = better (signal fidelity / lesion contrast, Eq. 18)")
    lines.append("  - ABLE uses learned apodization weights (θ) trained on synthetic data")
    lines.append("  - MVDR is the classical adaptive beamformer ABLE approximates")
    lines.append("    (Luijten et al. Sec. II-C: spatial smoothing + diagonal loading)")
    lines.append("  - B-mode images show typical ultrasound (dB-compressed, log scale)")
    lines.append("=" * 70)

    return "\n".join(lines)


def _apply_mm_axes(ax, nz, nx, n_ticks=5):
    """Label a B-mode panel's axes in mm, matching ForwardModel's imaging
    grid (lateral +/-X_LIM_MM, depth Z_START_MM..Z_END_MM) — the way axes
    are labeled in the paper's figures. Ticks are placed by explicit pixel
    index rather than relying on imshow's extent parameter, so the
    labeling is correct regardless of imshow's internal origin handling:
    row 0 (displayed at the top, imshow's default origin='upper') is the
    shallowest depth; row nz-1 (bottom) is the deepest.
    """
    x_pos = np.linspace(0, nx - 1, n_ticks)
    x_lab = np.linspace(-X_LIM_MM, X_LIM_MM, n_ticks)
    z_pos = np.linspace(0, nz - 1, n_ticks)
    z_lab = np.linspace(Z_START_MM, Z_END_MM, n_ticks)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{v:.0f}" for v in x_lab], fontsize=6)
    ax.set_yticks(z_pos)
    ax.set_yticklabels([f"{v:.0f}" for v in z_lab], fontsize=6)
    ax.tick_params(length=2, pad=1)


def save_visualizations(results, output_dir='demo_output', dyn_range=60,
                        model=None, max_points=20, forward_name='Analytic',
                        elem_dropout=0.0):
    """Save B-mode images as PNG (requires matplotlib).

    dyn_range: display dynamic range in dB (vmin = -dyn_range). Default 40,
    chosen from measurement, not taste: the weakest genuine scatterer
    response sits at -13..-21 dB while the DAS/MVDR electronic-noise floor
    sits at -42..-44 dB — a 40 dB window keeps every real echo >=19 dB
    above the display floor while pushing the noise wall just below it
    (its upper tail stays faintly visible). Metrics are computed on the
    raw maps and are unaffected by this display choice.

    Layout: panels wrap to a 2-row grid once there are more than 4 (e.g.
    GT + 7 methods with both ABLE++ and ABLE++S = 8 panels -> 2 rows of 4),
    instead of one ever-widening row. Every panel gets numeric axis ticks
    in mm; the "Lateral (mm)" / "Depth (mm)" axis titles are shown only on
    the bottom row / left column to avoid repeating the same text 8 times.

    model (if given) adds a second caption line with the imaging grid,
    array/physics parameters, and acquisition settings shared by every
    panel — so the figure states what was reconstructed and under what
    conditions, not just each method's metrics.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualization")
        return

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    panel_keys = ['gt'] + METHOD_KEYS
    n_panels = len(panel_keys)
    if n_panels <= 4:
        nrows, ncols = 1, n_panels
    else:
        ncols = int(np.ceil(n_panels / 2))
        nrows = 2

    info_suffix = ""
    if model is not None:
        info1, info2 = _imaging_info_lines(model, max_points, forward_name, elem_dropout)
        info_suffix = f"\n{info1}\n{info2}"

    for res in results:
        case = res['case']
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 5.3 * nrows))
        axes = np.asarray(axes).reshape(nrows, ncols)
        fig.suptitle(f"Test Case {case} ({res['type']}): B-mode Reconstruction "
                     f"(dB, common scale, {dyn_range:g} dB display range)"
                     f"{info_suffix}",
                     fontsize=12, y=1.06 if info_suffix else 1.02)

        nz, nx = res['gt_bmode'].shape

        for i, key in enumerate(panel_keys):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            if key == 'gt':
                ax.imshow(res['gt_bmode'], cmap='gray', vmin=-dyn_range, vmax=0)
                ax.set_title(f"Ground Truth\n{res['n_scatterers']} scatterers", fontsize=10)
            else:
                m = res['methods'][key]
                ax.imshow(m['bmode'], cmap='gray', vmin=-dyn_range, vmax=0)
                ax.set_title(f"{METHOD_NAMES[key]}\n"
                             f"MAE {m['mae']:.4f} | PSNR {m['psnr']:.1f} dB\n"
                             f"SMSLE {m['smsle']:.3f} | CNR {m['cnr']:.1f} dB",
                             fontsize=9)
            _apply_mm_axes(ax, nz, nx)
            if c == 0:
                ax.set_ylabel("Depth (mm)", fontsize=8)
            if r == nrows - 1:
                ax.set_xlabel("Lateral (mm)", fontsize=8)

        for j in range(n_panels, nrows * ncols):     # hide unused trailing slots
            r, c = divmod(j, ncols)
            axes[r, c].axis('off')

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
            'gt': torch.from_numpy(res['gt']),
            **{k: torch.from_numpy(res['methods'][k]['norm']) for k in METHOD_KEYS},
            'metrics': {
                k: {metric: res['methods'][k][metric]
                    for metric in ('mae', 'psnr', 'smsle', 'cnr')}
                for k in METHOD_KEYS
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
    parser.add_argument('--n_test', type=int, default=6,
                        help='number of test cases to run')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--M', type=int, default=64)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='demo_output',
                        help='directory to save results')
    parser.add_argument('--max_points', type=int, default=20,
                        help='cap on scatterers per test case (keeps images '
                             'visually countable; dense cases become '
                             'max_points isolated points)')
    parser.add_argument('--pp_checkpoint', type=str, default="/home/taah3149/Documents/Research Project/checkpoints_pp5/checkpoint_latest.pt",
                        help='ABLE++ checkpoint (trained with --learn_tx) — '
                             'adds a joint TX+RX column on the same cases')
    parser.add_argument('--pps_checkpoint', type=str, default="/home/taah3149/Documents/Research Project/checkpoints_pp6_smooth/checkpoint_latest.pt",
                        help='ABLE++S checkpoint (--learn_tx --tx_smooth > 0) '
                             '— adds a SEPARATE smoothness-regularized joint '
                             'TX+RX column alongside (not instead of) ABLE++')
    parser.add_argument('--forward', type=str, default='analytic',
                        choices=['analytic', 'deepwave'],
                        help='forward model that simulates the test RF data')
    parser.add_argument('--dyn_range', type=float, default=40,
                        help='display dynamic range in dB for the B-mode '
                             'panels (metrics are unaffected); 40 keeps all '
                             'real echoes visible while hiding the -42 dB '
                             'DAS noise floor')
    parser.add_argument('--elem_dropout', type=float, default=0.0,
                        help='fraction of elements dead for ALL methods '
                             '(damaged-aperture evaluation); dead elements '
                             'neither transmit nor receive')
    parser.add_argument('--dropout_seed', type=int, default=0,
                        help='seed for the fixed dead-element mask')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Reconstruction model is always the analytic one (DAS/FISTA/MVDR/ABLE
    # operators). With --forward deepwave the TEST DATA comes from the
    # wave-equation simulator instead — a physics-mismatch evaluation.
    model = ForwardModel(M=args.M, nx=args.nx, nz=args.nz, device=device).to(device)
    sim_model = None
    if args.forward == 'deepwave':
        from able_plus_plus.physics.deepwave_model import DeepwaveForwardModel
        sim_model = DeepwaveForwardModel(M=args.M, nx=args.nx, nz=args.nz,
                                         device=device).to(device)
        print("Test RF data: Deepwave (finite-difference wave equation); "
              "reconstruction operators: analytic")

    # Load checkpoint
    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    mlp, _, pos = load_network(args.checkpoint, N=args.M * args.M,
                               device=device, nx=args.nx, nz=args.nz)
    mlp.eval()

    pp = None
    if args.pp_checkpoint:
        if not Path(args.pp_checkpoint).exists():
            print(f"ERROR: ABLE++ checkpoint not found at {args.pp_checkpoint}")
            sys.exit(1)
        mlp_pp, ckpt_pp, pos_pp = load_network(args.pp_checkpoint,
                                               N=args.M * args.M, device=device,
                                               nx=args.nx, nz=args.nz)
        if 'tx_state' not in ckpt_pp:
            print("ERROR: --pp_checkpoint has no tx_state — was it trained "
                  "with --learn_tx?")
            sys.exit(1)
        tx = TxParams(M=args.M, fc=model.fc, fs=model.fs,
                      max_delay_periods=ckpt_pp.get('config', {})
                                              .get('tx_max_delay_periods', 1.0)
                      ).to(device)
        tx.load_state_dict(ckpt_pp['tx_state'])
        mlp_pp.eval()
        METHOD_KEYS.append('able_pp')
        pp = (mlp_pp, tx, pos_pp)
        print(f"Loaded ABLE++ TX params from {args.pp_checkpoint}")

    pps = None
    if args.pps_checkpoint:
        if not Path(args.pps_checkpoint).exists():
            print(f"ERROR: ABLE++S checkpoint not found at {args.pps_checkpoint}")
            sys.exit(1)
        mlp_pps, ckpt_pps, pos_pps = load_network(args.pps_checkpoint,
                                                  N=args.M * args.M, device=device,
                                                  nx=args.nx, nz=args.nz)
        if 'tx_state' not in ckpt_pps:
            print("ERROR: --pps_checkpoint has no tx_state — was it trained "
                  "with --learn_tx?")
            sys.exit(1)
        tx_pps = TxParams(M=args.M, fc=model.fc, fs=model.fs,
                          max_delay_periods=ckpt_pps.get('config', {})
                                                    .get('tx_max_delay_periods', 1.0)
                          ).to(device)
        tx_pps.load_state_dict(ckpt_pps['tx_state'])
        mlp_pps.eval()
        METHOD_KEYS.append('able_pps')
        pps = (mlp_pps, tx_pps, pos_pps)
        print(f"Loaded ABLE++S TX params from {args.pps_checkpoint} "
              f"(tx_smooth={ckpt_pps.get('config', {}).get('tx_smooth', 'unknown')})")

    # Fixed dead-element mask for the damaged-aperture evaluation
    elem_mask = None
    if args.elem_dropout > 0:
        g = torch.Generator().manual_seed(args.dropout_seed)
        elem_mask = (torch.rand(args.M, generator=g) >= args.elem_dropout
                     ).float().to(device)
        print(f"Damaged aperture: {int((1 - elem_mask).sum())}/{args.M} "
              f"elements dead (seed {args.dropout_seed})")

    # Run comparisons
    print(f"\nRunning {args.n_test} test cases...")
    results = compare_methods(model, mlp, n_test_cases=args.n_test,
                              device=device, max_points=args.max_points,
                              pp=pp, pps=pps, sim_model=sim_model, pos=pos,
                              elem_mask=elem_mask)

    # Generate report
    forward_name = 'Deepwave (FD wave-equation)' if args.forward == 'deepwave' else 'Analytic'
    report = format_report(results, model, args.max_points, forward_name,
                           elem_dropout=args.elem_dropout)
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

    save_visualizations(results, args.output_dir, dyn_range=args.dyn_range,
                       model=model, max_points=args.max_points,
                       forward_name=forward_name, elem_dropout=args.elem_dropout)

    print(f"\nDone. Results in {out_dir}/")


if __name__ == '__main__':
    main()
