"""Plot training progress from a run's checkpoint — works mid-training.

Reads the per-epoch history stored in checkpoint_latest.pt (updated every
epoch by run_training.py) and writes loss curves next to it. Re-run any
time to refresh the figure while training is going.

Usage:
    python scripts/plot_training.py                            # Trained_Checkpoints/checkpoints/
    python scripts/plot_training.py --checkpoint_dir Trained_Checkpoints/checkpoints_pp5
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint_dir', type=str, default='Trained_Checkpoints/checkpoints')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='explicit checkpoint path (overrides --checkpoint_dir)')
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint or Path(args.checkpoint_dir) / 'checkpoint_latest.pt')
    if not ckpt_path.exists():
        sys.exit(f"No checkpoint at {ckpt_path} yet — wait for the first epoch to finish.")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hist = ckpt['history']
    if not hist:
        sys.exit(f"{ckpt_path} has an empty history.")

    arch   = ckpt.get('config', {}).get('arch', 'mlp')
    epochs = [h['epoch'] for h in hist]
    loss   = [h['loss'] for h in hist]
    img    = [h['image_loss'] for h in hist]
    unity  = [h['unity_loss'] for h in hist]
    val    = [h['val_loss'] for h in hist]

    print(f"{ckpt_path}  (arch={arch})")
    print(f"history: {len(hist)} epochs (epoch {epochs[0]}..{epochs[-1]})")
    print(f"latest: loss={loss[-1]:.4f} img={img[-1]:.4f} "
          f"unity={unity[-1]:.6f} val={val[-1]:.4f}")
    print(f"best:   loss={min(loss):.4f} (epoch {epochs[int(np.argmin(loss))]})")

    C = {'total': '#1f77b4', 'img': '#2ca02c', 'unity': '#d62728', 'val': '#9467bd'}

    def plot_curves(ax, logscale):
        ax.plot(epochs, loss, color=C['total'], lw=1.8, label='train total')
        ax.plot(epochs, img, color=C['img'], lw=1.2, alpha=0.8, label='train image (SMSLE)')
        ax.plot(epochs, unity, color=C['unity'], lw=1.2, alpha=0.8, label='train unity-gain')
        ax.plot(epochs, val, color=C['val'], lw=1.2, ls='--', alpha=0.9, label='validation total')
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss' + (' (log scale)' if logscale else ''))
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_curves(axes[0], logscale=False)
    axes[0].set_title('linear scale')
    plot_curves(axes[1], logscale=True)
    axes[1].set_title('log scale')
    fig.suptitle(f'ABLE ({arch.upper()}) training — epoch {epochs[-1]}', fontsize=14)
    fig.tight_layout()

    out = ckpt_path.parent / 'training_progress.png'
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
