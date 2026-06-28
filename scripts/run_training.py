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
    torch.save({
        'step':            step,
        'model_state':     model.state_dict(),
        'mlp_state':       mlp.state_dict(),
        'optimizer_state': opt.state_dict(),
        'history':         history[-200:],
        'config':          cfg,
        'best_loss':       best_loss,
        'saved_at':        datetime.now().isoformat(),
    }, ckpt_path)
    logging.info(f"Checkpoint saved -> {ckpt_path}")


def load_checkpoint(ckpt_path, model, mlp, opt, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    mlp.load_state_dict(ckpt['mlp_state'])
    opt.load_state_dict(ckpt['optimizer_state'])
    logging.info(f"Resumed from {ckpt_path}  (step {ckpt['step']})")
    return ckpt['step'], ckpt.get('history', []), ckpt.get('best_loss', float('inf'))


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


# ============================================================
# training loop
# ============================================================

def run_training(n_steps, model, mlp, opt, cfg, device,
                 history, best_loss, ckpt_path, status_path,
                 start_step=0):

    log_every  = cfg['log_interval']
    ckpt_every = cfg['checkpoint_interval']
    t0 = time.time()

    for i in range(1, n_steps + 1):
        global_step = start_step + i

        l_total, l_img, l_unity = train_step(model, mlp, opt, cfg, device)
        history.append({
            'step':        global_step,
            'loss':        l_total,
            'image_loss':  l_img,
            'unity_loss':  l_unity,
        })

        if l_total < best_loss:
            best_loss = l_total

        elapsed = time.time() - t0

        if i % log_every == 0 or i == n_steps:
            logging.info(
                f"step {i:>{len(str(n_steps))}}/{n_steps}"
                f"  loss={l_total:.4f}  img={l_img:.4f}  unity={l_unity:.4f}"
                f"  best={best_loss:.4f}  ETA={eta_str(elapsed, i, n_steps)}"
            )
            write_status(status_path,
                step=global_step,
                total_steps=start_step + n_steps,
                stage_step=i,
                stage_total=n_steps,
                loss=round(l_total, 6),
                image_loss=round(l_img, 6),
                unity_loss=round(l_unity, 6),
                best_loss=round(best_loss, 6),
                elapsed_s=round(elapsed, 1),
                eta=eta_str(elapsed, i, n_steps),
                updated=datetime.now().isoformat(),
            )

        if i % ckpt_every == 0 or i == n_steps:
            save_checkpoint(ckpt_path, global_step,
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
    p.add_argument('--log_interval',    type=int,   default=20)
    p.add_argument('--ckpt_interval',   type=int,   default=200)
    p.add_argument('--checkpoint_dir',  type=str,   default='checkpoints')
    p.add_argument('--resume',          type=str,   default=None)
    p.add_argument('--dropout',         type=float, default=0.2)
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
        'log_interval':         args.log_interval,
        'checkpoint_interval':  args.ckpt_interval,
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
    start_step  = 0

    resume_path = args.resume or (ckpt_path if ckpt_path.exists() else None)
    if resume_path and Path(resume_path).exists():
        start_step, history, best_loss = load_checkpoint(resume_path, model, mlp, opt, device)

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
        start_step=start_step,
    )

    logging.info(f"Training complete.  Best loss: {best_loss:.4f}")
    logging.info(f"Final checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
