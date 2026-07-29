"""Visualize the learned global transmit parameters of ABLE++.

Reads tx_state from a --learn_tx checkpoint and plots the two per-element
vectors (applied on the transmit side by simulate_tx):

    w_tx   [M]  transmit apodization, tanh-bounded to (-1, 1)
    tau_tx [M]  transmit firing delays, tanh-bounded to +/- max_delay_periods
                carrier periods (fs/fc samples each)

Usage:
    python scripts/plot_tx_params.py [--checkpoint checkpoints_pp/checkpoint_latest.pt]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from able_plus_plus.networks.tx_params import TxParams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=str,
                   default='checkpoints_pp/checkpoint_latest.pt')
    p.add_argument('--M', type=int, default=64)
    p.add_argument('--fc', type=float, default=5e6)
    p.add_argument('--fs', type=float, default=40e6)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'tx_state' not in ckpt:
        sys.exit(f"{args.checkpoint} has no tx_state (train with --learn_tx)")

    n_periods = ckpt.get('config', {}).get('tx_max_delay_periods', 1.0)
    tx = TxParams(M=args.M, fc=args.fc, fs=args.fs, max_delay_periods=n_periods)
    tx.load_state_dict(ckpt['tx_state'])
    with torch.no_grad():
        w, tau = tx()
    max_delay = tx.max_delay
    period = tx.period_samples

    elems = range(args.M)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].axhline(0.0, color='grey', lw=1, ls='--', label='w=0 (silent)')
    axes[0].plot(elems, w.numpy(), color='#177E8C', lw=1.8, marker='.', ms=5)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_xlabel('transmit element index')
    axes[0].set_ylabel('$w_{tx}$')
    axes[0].set_title('Transmit apodization (bounded to (-1, 1))', fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].axhline(0.0, color='grey', lw=1, ls='--', label='geometric timing')
    for b in (max_delay, -max_delay):
        axes[1].axhline(b, color='grey', lw=0.8, ls=':')
    axes[1].plot(elems, tau.numpy(), color='#A9641C', lw=1.8, marker='.', ms=5)
    axes[1].set_ylim(-1.15 * max_delay, 1.15 * max_delay)
    axes[1].set_xlabel('transmit element index')
    axes[1].set_ylabel(r'$\tau_{tx}$ (samples)')
    axes[1].set_title(f'Transmit firing delays (bounded to ±{n_periods:g} '
                      f'period{"s" if n_periods != 1 else ""} = '
                      f'±{max_delay:.1f} samples)', fontsize=11)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=0.25)

    fig.suptitle('ABLE++ learned transmit parameters (global, per element, '
                 'randomly initialized, applied on transmit)', fontsize=13)
    fig.tight_layout()
    out = Path(args.checkpoint).parent / 'tx_params.png'
    fig.savefig(out, dpi=120)
    print(f"carrier period: {period:.2f} samples (fs/fc)")
    print(f"w_tx  range: {w.min():.3f} .. {w.max():.3f}")
    print(f"tau   range: {tau.min():.2f} .. {tau.max():.2f} samples "
          f"(bound ±{max_delay:.2f} = ±{n_periods:g} period(s))")
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
