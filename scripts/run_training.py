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
from able_plus_plus.networks.tx_params import TxParams
from able_plus_plus.data.simulate import make_batch, random_scatterer_batch
from able_plus_plus.networks.able_mlp import apply_mlp, pixel_positions
from able_plus_plus.networks.losses import total_loss, tx_smoothness_loss


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


def save_checkpoint(ckpt_path, step, model, mlp, opt, history, cfg, best_loss,
                    tx=None):
    # model_state is intentionally NOT saved: the ForwardModel has no
    # parameters, only ~1.7 GB of geometry buffers that are rebuilt
    # deterministically from the config in __init__. Saving them made every
    # checkpoint 1.8 GB and blew the home-directory disk quota.
    payload = {
        'step':            step,
        'mlp_state':       mlp.state_dict(),
        'optimizer_state': opt.state_dict(),
        'history':         history[-200:],
        'config':          cfg,
        'best_loss':       best_loss,
        'saved_at':        datetime.now().isoformat(),
    }
    if tx is not None:
        payload['tx_state'] = tx.state_dict()
    torch.save(payload, ckpt_path)
    logging.info(f"Checkpoint saved -> {ckpt_path}")


def load_checkpoint(ckpt_path, model, mlp, opt, device, epoch_size, tx=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state' in ckpt:   # old-format checkpoints stored the buffers
        model.load_state_dict(ckpt['model_state'])
    mlp.load_state_dict(ckpt['mlp_state'])
    opt.load_state_dict(ckpt['optimizer_state'])
    if tx is not None and 'tx_state' in ckpt:
        tx.load_state_dict(ckpt['tx_state'])
    # 'step' stores the completed EPOCH count (save_checkpoint is called
    # with start_epoch + epoch) — do NOT divide by epoch_size; doing so
    # restarted resumed runs at epoch 0 with a reset LR schedule.
    epoch = ckpt.get('step', 0)
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

def acquire(model, gt_images, cfg, tx=None, noise=None):
    """Simulate the acquisition for a batch of ground-truth maps.

    Receive-only mode (tx=None): the frozen physics under no_grad, exactly
    the paper's setup.

    Joint ABLE++ mode: the LEARNED transmit apodization + firing delays are
    applied on the simulator side (simulate_tx), i.e. the acquisition
    itself is part of the computational graph and gradients flow from the
    loss back through the physics into w_tx / tau_tx.

    Optional robustness augmentations — both applied on the TRANSMIT side,
    via simulate_tx, never in the reconstruction:
      domain_rand:  per-sample jitter of element gains (±5%) and firing
                    times (±0.5 samples) around the (learned or nominal)
                    values — trains tolerance to the physics-model errors
                    a real probe has (the deepwave mismatch, in spirit).
      elem_dropout: probability of each element being DEAD for this batch —
                    it neither transmits (w=0) nor receives (its rf channel
                    is zeroed). Trains the model to image with a damaged /
                    sparse aperture, where a learned firing strategy can
                    genuinely outperform uniform DAS.
    """
    dev = gt_images.device
    M = model.M
    w_tx = tau_tx = None
    if tx is not None:
        w_tx, tau_tx = tx()

    rand  = cfg.get('domain_rand', False)
    pdrop = cfg.get('elem_dropout', 0.0)
    mask = None
    if rand or pdrop > 0:
        if w_tx is None:
            w_tx = torch.ones(M, device=dev)
        if tau_tx is None:
            tau_tx = torch.zeros(M, device=dev)
        if rand:
            w_tx   = w_tx * (1.0 + 0.05 * torch.randn(M, device=dev))
            tau_tx = tau_tx + 0.5 * torch.randn(M, device=dev)
        if pdrop > 0:
            mask = (torch.rand(M, device=dev) >= pdrop).float()
            w_tx = w_tx * mask

    if w_tx is None and tau_tx is None:
        with torch.no_grad():
            rf = model.simulate(gt_images)
    elif tx is not None:
        rf = model.simulate_tx(gt_images, w_tx=w_tx, tau_tx=tau_tx)
    else:
        with torch.no_grad():
            rf = model.simulate_tx(gt_images, w_tx=w_tx, tau_tx=tau_tx)

    if mask is not None:                     # dead elements don't receive
        rf = rf * mask.view(1, M, 1)
    if noise is None:
        noise = torch.randn_like(rf)
    return rf + noise * cfg['noise_level'] * rf.detach().abs().max()


def train_step(model, mlp, opt, cfg, device, tx=None, pos=None):
    """Single optimisation step.

    Backprop reaches the MLP weights and, when tx is given, the transmit
    apodization/delay vectors — all three are updated together by the same
    optimizer against the same loss (joint TX+RX learning). In joint mode
    the gradient path for the TX side runs THROUGH the simulator:
    loss -> reconstruction -> das_adjoint -> rf_data -> simulate_tx -> {w_tx, tau_tx}.
    """
    opt.zero_grad()

    gt_images = random_scatterer_batch(cfg['batch_size'], model.nx, model.nz,
                                       scatterer_type=cfg['scatterer_type'],
                                       device=device)
    rf_data = acquire(model, gt_images, cfg, tx=tx)

    _, pre_summed = model.das_adjoint(rf_data)          # frozen reconstruction
    p_recon, weights, _ = apply_mlp(mlp, pre_summed, pos=pos)
    target = gt_images.detach()

    loss, l_img, l_unity = total_loss(p_recon, target, weights,
                                      lam=cfg['lam'],
                                      lam_bg=cfg.get('lam_bg', 0.0))
    if tx is not None and cfg.get('tx_smooth', 0.0) > 0:
        w_tx, tau_tx = tx()
        loss = loss + cfg['tx_smooth'] * tx_smoothness_loss(w_tx, tau_tx, tx.max_delay)
    loss.backward()
    params = list(mlp.parameters()) + (list(tx.parameters()) if tx is not None else [])
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    opt.step()

    return float(loss), float(l_img), float(l_unity)


def eval_on_fixed_batch(model, mlp, fixed_data, cfg, device, tx=None, pos=None):
    """Evaluate on fixed validation data.

    In joint mode the acquisition depends on the CURRENT tx parameters, so
    the RF must be re-simulated each evaluation (only the ground truth and
    the noise realization stay fixed). Validation uses the CLEAN nominal
    acquisition (no domain_rand / elem_dropout jitter) so the curve
    measures the model, not the augmentation draw."""
    gt_images, rf_static, unit_noise = fixed_data

    val_cfg = dict(cfg, domain_rand=False, elem_dropout=0.0)
    with torch.no_grad():
        if tx is None:
            rf = rf_static
        else:
            rf = acquire(model, gt_images, val_cfg, tx=tx, noise=unit_noise)
        _, pre_summed = model.das_adjoint(rf)
        p_recon, weights, _ = apply_mlp(mlp, pre_summed, pos=pos)
        loss, l_img, l_unity = total_loss(p_recon, gt_images, weights,
                                          lam=cfg['lam'],
                                          lam_bg=cfg.get('lam_bg', 0.0))
        if tx is not None and cfg.get('tx_smooth', 0.0) > 0:
            w_tx, tau_tx = tx()
            loss = loss + cfg['tx_smooth'] * tx_smoothness_loss(w_tx, tau_tx, tx.max_delay)

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
                 start_epoch=0, fixed_val_data=None, out_dir=None, tx=None,
                 pos=None):

    epoch_size = cfg['epoch_size']
    # n_steps is the TOTAL budget for the run: a resumed job trains only the
    # remaining epochs, not n_steps more. (The CNN run resumed at epoch 38
    # and did 100 further epochs — 138 total — before this fix.)
    n_epochs = n_steps // epoch_size
    remaining = max(n_epochs - start_epoch, 0)
    if remaining == 0:
        logging.info(f"Nothing to do: checkpoint already at epoch "
                     f"{start_epoch}/{n_epochs}.")
        return history, best_loss
    t0 = time.time()

    if out_dir is None:
        out_dir = Path(ckpt_path).parent

    total_epochs = n_epochs
    lr0 = cfg['lr']
    lr_min = lr0 * 0.01

    for epoch in range(1, remaining + 1):
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
            l_total, l_img, l_unity = train_step(model, mlp, opt, cfg, device, tx=tx, pos=pos)
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
            val_loss, val_img, val_unity = eval_on_fixed_batch(model, mlp, fixed_val_data, cfg, device, tx=tx, pos=pos)

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
            f"Epoch {start_epoch + epoch:>{len(str(n_epochs))}}/{n_epochs}  "
            f"loss={avg_loss:.4f}  img={avg_img_loss:.4f}  unity={avg_unity_loss:.6f}  "
            f"best={best_loss:.4f}{val_str}  lr={lr:.2e}  ETA={eta_str(elapsed, epoch, remaining)}"
        )

        # Stop only on fatal anomalies (NaN/Inf); warn otherwise
        if is_fatal:
            logging.error(f"⚠️  LOSS ANOMALY DETECTED: {anomaly_msg}")
            logging.error(f"⚠️  Training stopped at epoch {epoch}/{n_epochs}")
            logging.error(f"⚠️  Saved emergency checkpoint before stopping")
            save_checkpoint(out_dir / 'checkpoint_emergency.pt',
                           start_epoch + epoch, model, mlp, opt, history, cfg,
                           best_loss, tx=tx)
            raise RuntimeError(f"Training stopped due to loss anomaly: {anomaly_msg}")
        elif anomaly_msg:
            logging.warning(f"Loss anomaly (non-fatal, continuing): {anomaly_msg}")

        write_status(status_path,
            epoch=start_epoch + epoch,
            total_epochs=n_epochs,
            loss=round(avg_loss, 6),
            image_loss=round(avg_img_loss, 6),
            unity_loss=round(avg_unity_loss, 6),
            val_loss=round(val_loss, 6) if val_loss is not None else None,
            best_loss=round(best_loss, 6),
            elapsed_s=round(elapsed, 1),
            eta=eta_str(elapsed, epoch, remaining),
            updated=datetime.now().isoformat(),
        )

        # Checkpoint every 2 epochs
        if epoch % 2 == 0 or epoch == remaining:
            save_checkpoint(ckpt_path, start_epoch + epoch,
                            model, mlp, opt, history, cfg, best_loss, tx=tx)

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
    p.add_argument('--learn_tx',        action='store_true',
                   help='ABLE++ joint mode: learn global per-element '
                        'transmit apodization + firing delays (applied on '
                        'the simulator side, simulate_tx) together with the '
                        'receive MLP (same optimizer, same loss)')
    p.add_argument('--tx_max_delay_periods', type=float, default=1.0,
                   help='bound for the learned transmit firing delays, in '
                        'PERIODS of the carrier sinusoid (fs/fc samples '
                        'each) — tanh-limited so shifts stay inside the '
                        'pulse support and gradients stay alive')
    p.add_argument('--tx_seed',         type=int,   default=None,
                   help='seed for the random TX parameter initialization '
                        '(default: nondeterministic)')
    p.add_argument('--pos_enc',         action='store_true',
                   help='append (x, z) positional features to the MLP '
                        'input — position-aware receive apodization '
                        '(the spatial awareness Luijten et al. Sec. VI '
                        'names as future work)')
    p.add_argument('--lam_bg',          type=float, default=0.0,
                   help='weight of the linear-domain background-power '
                        'loss term (aligns training with CNR/PSNR); '
                        '0 = paper objective unchanged')
    p.add_argument('--domain_rand',     action='store_true',
                   help='per-batch transmit-side jitter (element gains '
                        '±5%%, firing times ±0.5 smp) for physics-mismatch '
                        'robustness')
    p.add_argument('--elem_dropout',    type=float, default=0.0,
                   help='per-batch probability of each element being dead '
                        '(neither transmits nor receives) — damaged/sparse '
                        'aperture training')
    p.add_argument('--tx_smooth',       type=float, default=0.0,
                   help='weight of the total-variation smoothness penalty '
                        'on w_tx/tau_tx across element index — discourages '
                        'the jagged, high-sidelobe aperture weighting an '
                        'unconstrained per-element TX solution tends to '
                        'find (0 = off, the ABLE++ behavior without this fix)')
    # NOTE: training always uses the analytic forward model. Deepwave is a
    # TEST-time physics-mismatch simulator (demo_results.py --forward
    # deepwave), not a training data generator.
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
        'learn_tx':             args.learn_tx,
        'tx_max_delay_periods': args.tx_max_delay_periods,
        'pos_enc':              args.pos_enc,
        'lam_bg':               args.lam_bg,
        'domain_rand':          args.domain_rand,
        'elem_dropout':         args.elem_dropout,
        'tx_smooth':            args.tx_smooth,
    }

    model = ForwardModel(M=args.M, nx=args.nx, nz=args.nz, device=device).to(device)
    mlp = ABLEMLP(N=args.M * args.M, dropout=args.dropout,
                  pos_dim=2 if args.pos_enc else 0).to(device)
    pos = pixel_positions(args.nx, args.nz, device) if args.pos_enc else None

    tx = None
    if args.learn_tx:
        tx = TxParams(M=args.M, fc=model.fc, fs=model.fs,
                      max_delay_periods=args.tx_max_delay_periods,
                      seed=args.tx_seed).to(device)

    n_mlp = sum(p.numel() for p in mlp.parameters())
    logging.info(f"ForwardModel: M={args.M}, nx={args.nx}, nz={args.nz}")
    logging.info(f"ABLEMLP: input={args.M*args.M}  params={n_mlp:,}")
    if tx is not None:
        n_tx = sum(p.numel() for p in tx.parameters())
        logging.info(f"TxParams: global per-element, random init, |tau| < "
                     f"{args.tx_max_delay_periods:g} period(s) "
                     f"(={tx.max_delay:.1f} samples), params={n_tx:,} "
                     f"(joint ABLE++ mode, transmit-side)")
        if args.tx_smooth > 0:
            logging.info(f"TX smoothness penalty: lam_tx_smooth={args.tx_smooth:g} "
                         f"(ABLE++S mode)")

    params = list(mlp.parameters()) + (list(tx.parameters()) if tx else [])
    opt = torch.optim.Adam(params, lr=args.lr)

    ckpt_path   = out_dir / 'checkpoint_latest.pt'
    status_path = out_dir / 'status.json'
    history     = []
    best_loss   = float('inf')
    start_epoch = 0

    resume_path = args.resume or (ckpt_path if ckpt_path.exists() else None)
    if resume_path and Path(resume_path).exists():
        start_epoch, history, best_loss = load_checkpoint(
            resume_path, model, mlp, opt, device, args.epoch_size, tx=tx)

    # Generate fixed validation data. In joint-TX mode the acquisition
    # depends on the current learned parameters, so only the ground truth
    # and the noise realization are fixed; RF is re-simulated at eval time.
    logging.info("Generating fixed validation batch (8 samples)...")
    gt_val, rf_val = make_batch(
        model,
        batch_size=8,
        noise_level=cfg['noise_level'],
        scatterer_type='mixed',
        device=device,
    )
    fixed_val_data = (gt_val, rf_val, torch.randn_like(rf_val))
    logging.info("✓ Fixed validation data ready (8 samples)")

    def _sighandler(sig, frame):
        logging.warning(f"Signal {sig} received — saving emergency checkpoint")
        # (previously referenced an undefined 'start_step' and crashed the
        # handler, so no emergency checkpoint was actually written)
        last_epoch = history[-1]['epoch'] if history else start_epoch
        save_checkpoint(out_dir / 'checkpoint_emergency.pt',
                        last_epoch, model, mlp, opt, history, cfg, best_loss,
                        tx=tx)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sighandler)
    signal.signal(signal.SIGINT,  _sighandler)

    history, best_loss = run_training(
        args.n_steps, model, mlp, opt, cfg, device,
        history, best_loss, ckpt_path, status_path,
        start_epoch=start_epoch,
        fixed_val_data=fixed_val_data,
        out_dir=out_dir,
        tx=tx,
        pos=pos,
    )

    logging.info(f"Training complete.  Best loss: {best_loss:.4f}")
    logging.info(f"Final checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
