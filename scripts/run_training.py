"""Standalone training entry-point for ABLE++ on the Makalu HPC cluster.

Run via the LSF job script (submit_job.lsf) or locally:
    python run_training.py --n_steps 5000 --batch_size 4

Only the receive MLP (theta) is trained.  The forward model is a fixed
physics engine — no joint TX optimisation stage.

Saves checkpoints to <checkpoint_dir>/ and writes a machine-readable
status.json every --log_interval steps so that `python status.py` can
report live training progress from any login node.
"""
import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from able_plus_plus import ForwardModel, ABLEMLP
from able_plus_plus.data.simulate import make_batch
from able_plus_plus.networks.able_mlp import apply_mlp
from able_plus_plus.networks.losses import total_loss


# ============================================================
# helpers
# ============================================================

def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}"
                     f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // 2**20} MB")
        return dev
    logging.warning("No GPU found -- falling back to CPU (will be slow)")
    return torch.device('cpu')


def setup_logging(log_file):
    fmt = '%(asctime)s  %(levelname)-8s  %(message)s'
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode='a')]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def write_status(path, **kwargs):
    """Atomically write JSON status — status.py never reads a half-written file."""
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(kwargs, f, indent=2)
    os.replace(tmp, path)


def save_checkpoint(ckpt_path, step, model, mlp, opt, history, cfg, best_loss):
    # model_state is intentionally NOT saved: the ForwardModel has no
    # parameters, only ~1.7 GB of geometry buffers that are rebuilt
    # deterministically from the config in __init__. Saving them made every
    # checkpoint 1.8 GB and blew the home-directory disk quota.
    torch.save({
        'step':            step,
        'mlp_state':       mlp.state_dict(),
        'optimizer_state': opt.state_dict(),
        'history':         history[-200:],
        'config':          cfg,
        'best_loss':       best_loss,
        'saved_at':        datetime.now().isoformat(),
    }, ckpt_path)
    logging.info(f"Checkpoint saved -> {ckpt_path}")


def load_checkpoint(ckpt_path, model, mlp, opt, device, epoch_size):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state' in ckpt:   # old-format checkpoints stored the buffers
        model.load_state_dict(ckpt['model_state'])
    mlp.load_state_dict(ckpt['mlp_state'])
    opt.load_state_dict(ckpt['optimizer_state'])
    # Convert step to epoch
    step = ckpt.get('step', 0)
    epoch = step // epoch_size if step > 0 else 0
    logging.info(f"Resumed from {ckpt_path}  (epoch {epoch})")
    return epoch, ckpt.get('history', []), ckpt.get('best_loss', float('inf'))


def eta_str(elapsed, step, total_steps):
    if step == 0:
        return 'unknown'
    remaining = (elapsed / step) * (total_steps - step)
    return str(timedelta(seconds=int(remaining)))


# ============================================================
# one training step
# ============================================================

def train_step(model, mlp, opt, cfg, device):
    """Single optimisation step — backprop into MLP weights only."""
    opt.zero_grad()

    gt_images, rf_data = make_batch(
        model,
        cfg['batch_size'],
        noise_level=cfg['noise_level'],
        scatterer_type=cfg['scatterer_type'],
        device=device,
    )

    _, pre_summed = model.das_adjoint(rf_data)           # [B, M*M, P]
    p_recon, weights, _ = apply_mlp(mlp, pre_summed)    # [B, P], [B*P, M*M]
    target = gt_images.detach()

    loss, l_img, l_unity = total_loss(p_recon, target, weights, lam=cfg['lam'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
    opt.step()

    return float(loss), float(l_img), float(l_unity)


def eval_on_fixed_batch(model, mlp, fixed_data, cfg, device):
    """Evaluate on fixed validation data (same every time)."""
    gt_images, rf_data = fixed_data

    with torch.no_grad():
        _, pre_summed = model.das_adjoint(rf_data)
        p_recon, weights, _ = apply_mlp(mlp, pre_summed)
        loss, l_img, l_unity = total_loss(p_recon, gt_images, weights, lam=cfg['lam'])

    return float(loss), float(l_img), float(l_unity)


# ============================================================
# training loop
# ============================================================

def check_loss_sanity(history, current_epoch):
    """Detect anomalies in the loss trajectory.

    Returns: (is_fatal, message)
        is_fatal=True  -> training cannot continue (NaN/Inf)
        message set with is_fatal=False -> log a warning but keep training.

    Only non-finite loss is fatal: fast drops are EXPECTED behaviour early
    in training (especially right after a loss-function change), rises and
    plateaus are things to watch, not reasons to kill an HPC job.
    """
    if len(history) < 2:
        return False, None

    prev_loss = history[-2]['loss']
    curr_loss = history[-1]['loss']

    # Fatal: NaN or Inf
    if not (torch.isfinite(torch.tensor(curr_loss)).item()):
        return True, f"Loss is {curr_loss} (NaN/Inf detected)"

    # Warning: loss increasing rapidly (>200% increase)
    if curr_loss > prev_loss * 3:
        pct_increase = (curr_loss - prev_loss) / prev_loss * 100
        return False, f"Loss rising fast: {prev_loss:.4f} → {curr_loss:.4f} ({pct_increase:.1f}% increase)"

    # Warning: same loss for 3+ consecutive epochs (stuck)
    if len(history) >= 3:
        last_3_losses = [h['loss'] for h in history[-3:]]
        if abs(last_3_losses[0] - last_3_losses[1]) < 0.0001 and \
           abs(last_3_losses[1] - last_3_losses[2]) < 0.0001:
            return False, f"Loss stuck at {curr_loss:.6f} for 3+ epochs (no learning)"

    # Warning: train/val catastrophic mismatch (>30x)
    if history[-1].get('val_loss') is not None:
        val_loss = history[-1]['val_loss']
        if curr_loss > 0 and val_loss > 0:
            ratio = max(curr_loss, val_loss) / min(curr_loss, val_loss)
            if ratio > 30:
                return False, f"Training/validation mismatch: train={curr_loss:.4f}, val={val_loss:.4f} (ratio={ratio:.1f}x)"

    return False, None


def run_training(n_steps, model, mlp, opt, cfg, device,
                 history, best_loss, ckpt_path, status_path,
                 start_epoch=0, fixed_val_data=None, out_dir=None):

    epoch_size = cfg['epoch_size']
    n_epochs = n_steps // epoch_size
    t0 = time.time()

    if out_dir is None:
        out_dir = Path(ckpt_path).parent

    total_epochs = start_epoch + n_epochs
    lr0 = cfg['lr']
    lr_min = lr0 * 0.01

    for epoch in range(1, n_epochs + 1):
        # Cosine LR decay over the whole planned run (resume-safe: based on
        # the global epoch index). Constant lr=1e-3 kept the loss bouncing
        # in a wide band late in training; annealing to lr0/100 lets the
        # optimizer settle into a sharper optimum.
        g_epoch = start_epoch + epoch
        lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * (g_epoch - 1) / max(total_epochs - 1, 1)))
        for group in opt.param_groups:
            group['lr'] = lr

        epoch_losses = []
        epoch_img_losses = []
        epoch_unity_losses = []

        # Train for one epoch
        for step_in_epoch in range(epoch_size):
            l_total, l_img, l_unity = train_step(model, mlp, opt, cfg, device)
            epoch_losses.append(l_total)
            epoch_img_losses.append(l_img)
            epoch_unity_losses.append(l_unity)

        # Compute epoch averages
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_img_loss = sum(epoch_img_losses) / len(epoch_img_losses)
        avg_unity_loss = sum(epoch_unity_losses) / len(epoch_unity_losses)

        # Evaluate on fixed validation data
        val_loss = None
        val_img = None
        val_unity = None
        if fixed_val_data is not None:
            val_loss, val_img, val_unity = eval_on_fixed_batch(model, mlp, fixed_val_data, cfg, device)

        if avg_loss < best_loss:
            best_loss = avg_loss

        elapsed = time.time() - t0

        # Append to history
        history.append({
            'epoch':           start_epoch + epoch,
            'loss':            avg_loss,
            'image_loss':      avg_img_loss,
            'unity_loss':      avg_unity_loss,
            'val_loss':        val_loss,
            'val_image_loss':  val_img,
            'val_unity_loss':  val_unity,
        })

        # Sanity check for loss anomalies
        is_fatal, anomaly_msg = check_loss_sanity(history, epoch)

        # Log epoch results
        global_step = start_epoch * epoch_size + epoch * epoch_size
        val_str = f"  val_loss={val_loss:.4f}" if val_loss is not None else ""
        logging.info(
            f"Epoch {epoch:>{len(str(n_epochs))}}/{n_epochs}  "
            f"loss={avg_loss:.4f}  img={avg_img_loss:.4f}  unity={avg_unity_loss:.6f}  "
            f"best={best_loss:.4f}{val_str}  lr={lr:.2e}  ETA={eta_str(elapsed, epoch, n_epochs)}"
        )

        # Stop only on fatal anomalies (NaN/Inf); warn otherwise
        if is_fatal:
            logging.error(f"⚠️  LOSS ANOMALY DETECTED: {anomaly_msg}")
            logging.error(f"⚠️  Training stopped at epoch {epoch}/{n_epochs}")
            logging.error(f"⚠️  Saved emergency checkpoint before stopping")
            save_checkpoint(out_dir / 'checkpoint_emergency.pt',
                           start_epoch + epoch, model, mlp, opt, history, cfg, best_loss)
            raise RuntimeError(f"Training stopped due to loss anomaly: {anomaly_msg}")
        elif anomaly_msg:
            logging.warning(f"Loss anomaly (non-fatal, continuing): {anomaly_msg}")

        write_status(status_path,
            epoch=start_epoch + epoch,
            total_epochs=start_epoch + n_epochs,
            loss=round(avg_loss, 6),
            image_loss=round(avg_img_loss, 6),
            unity_loss=round(avg_unity_loss, 6),
            val_loss=round(val_loss, 6) if val_loss is not None else None,
            best_loss=round(best_loss, 6),
            elapsed_s=round(elapsed, 1),
            eta=eta_str(elapsed, epoch, n_epochs),
            updated=datetime.now().isoformat(),
        )

        # Checkpoint every 2 epochs
        if epoch % 2 == 0 or epoch == n_epochs:
            save_checkpoint(ckpt_path, start_epoch + epoch,
                            model, mlp, opt, history, cfg, best_loss)

    return history, best_loss


# ============================================================
# main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='Train ABLE++ receive MLP on the Makalu HPC')
    p.add_argument('--n_steps',         type=int,   default=5000)
    p.add_argument('--batch_size',      type=int,   default=4)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--lam',             type=float, default=0.8,
                   help='weight of image loss in L_total')
    p.add_argument('--noise_level',     type=float, default=0.05)
    p.add_argument('--scatterer_type',  type=str,   default='mixed',
                   choices=['sparse', 'dense', 'clustered', 'mixed'],
                   help='training data diversity (mixed = random per sample)')
    p.add_argument('--M',               type=int,   default=64)
    p.add_argument('--nx',              type=int,   default=128)
    p.add_argument('--nz',              type=int,   default=128)
    p.add_argument('--epoch_size',      type=int,   default=100,
                   help='training steps per epoch')
    p.add_argument('--checkpoint_dir',  type=str,   default='checkpoints')
    p.add_argument('--resume',          type=str,   default=None)
    p.add_argument('--dropout',         type=float, default=0.0,
                   help='keep 0: data is generated fresh each step (no '
                        'overfitting possible) and dropout breaks the '
                        'unity-gain constraint at eval time')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir / 'training.log')
    logging.info("=" * 60)
    logging.info(f"ABLE++ training started  {datetime.now().isoformat()}")
    logging.info(f"Args: {vars(args)}")

    device = get_device()

    cfg = {
        'batch_size':           args.batch_size,
        'lr':                   args.lr,
        'lam':                  args.lam,
        'noise_level':          args.noise_level,
        'scatterer_type':       args.scatterer_type,
        'epoch_size':           args.epoch_size,
    }

    model = ForwardModel(M=args.M, nx=args.nx, nz=args.nz, device=device).to(device)
    mlp   = ABLEMLP(N=args.M * args.M, dropout=args.dropout).to(device)

    n_mlp = sum(p.numel() for p in mlp.parameters())
    logging.info(f"ForwardModel: M={args.M}, nx={args.nx}, nz={args.nz}")
    logging.info(f"ABLEMLP: input={args.M*args.M}  params={n_mlp:,}")

    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    ckpt_path   = out_dir / 'checkpoint_latest.pt'
    status_path = out_dir / 'status.json'
    history     = []
    best_loss   = float('inf')
    start_epoch = 0

    resume_path = args.resume or (ckpt_path if ckpt_path.exists() else None)
    if resume_path and Path(resume_path).exists():
        start_epoch, history, best_loss = load_checkpoint(resume_path, model, mlp, opt, device, args.epoch_size)

    # Generate fixed validation data
    logging.info("Generating fixed validation batch (8 samples)...")
    fixed_val_data = make_batch(
        model,
        batch_size=8,
        noise_level=cfg['noise_level'],
        scatterer_type='mixed',
        device=device,
    )
    logging.info("✓ Fixed validation data ready (8 samples)")

    def _sighandler(sig, frame):
        logging.warning(f"Signal {sig} received — saving emergency checkpoint")
        save_checkpoint(out_dir / 'checkpoint_emergency.pt',
                        start_step, model, mlp, opt, history, cfg, best_loss)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sighandler)
    signal.signal(signal.SIGINT,  _sighandler)

    history, best_loss = run_training(
        args.n_steps, model, mlp, opt, cfg, device,
        history, best_loss, ckpt_path, status_path,
        start_epoch=start_epoch,
        fixed_val_data=fixed_val_data,
        out_dir=out_dir,
    )

    logging.info(f"Training complete.  Best loss: {best_loss:.4f}")
    logging.info(f"Final checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
