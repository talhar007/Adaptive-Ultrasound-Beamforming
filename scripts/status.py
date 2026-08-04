"""Training status monitor for ABLE++.

Run this at any time from any login node to see the current state of your
running (or most-recently-finished) training job:

    python scripts/status.py                                       # Trained_Checkpoints/checkpoints/
    python scripts/status.py --checkpoint_dir Trained_Checkpoints/checkpoints_debug_das_adjoint_fix
    watch -n 30 python scripts/status.py --checkpoint_dir <dir>     # refresh every 30 seconds live

Reads <checkpoint_dir>/status.json (written by run_training.py every epoch)
and <checkpoint_dir>/training.log (last few lines).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


STALE_WARN = 120   # seconds before we warn the job may have died


def bar(current, total, width=40):
    filled = int(width * current / max(total, 1))
    return '[' + '#' * filled + '-' * (width - filled) + ']'


def fmt_time(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint_dir', type=str, default='Trained_Checkpoints/checkpoints',
                   help='directory a run_training.py job is writing to')
    args = p.parse_args()

    ckpt_dir    = Path(args.checkpoint_dir)
    status_file = ckpt_dir / 'status.json'
    log_file    = ckpt_dir / 'training.log'

    # ---- status.json ----
    if not status_file.exists():
        print(f"No status file found at {status_file}")
        print("Either training hasn't started yet, or --checkpoint_dir is wrong.")
        sys.exit(0)

    with open(status_file) as f:
        s = json.load(f)

    updated = datetime.fromisoformat(s['updated'])
    age     = (datetime.now() - updated).total_seconds()

    epoch       = s.get('epoch', 0)
    total_epoch = s.get('total_epochs', 1)
    pct         = 100 * epoch / max(total_epoch, 1)
    progress    = bar(epoch, total_epoch)

    print()
    print("=" * 60)
    print(f"  ABLE++ Training Status  ({ckpt_dir})")
    print("=" * 60)
    print(f"  Progress: {progress} {pct:5.1f}%  (epoch {epoch}/{total_epoch})")
    print()
    print(f"  Loss (total)    : {s.get('loss',        float('nan')):.6f}")
    print(f"  Loss (image)    : {s.get('image_loss',  float('nan')):.6f}")
    print(f"  Loss (unity)    : {s.get('unity_loss',  float('nan')):.6f}")
    print(f"  Weight magnitude: {s.get('weight_mag',  float('nan')):.4f}"
          "  (diagnostic -- unbounded canceling weights would show up here)")
    val_loss = s.get('val_loss')
    print(f"  Val loss        : {val_loss:.6f}" if val_loss is not None else "  Val loss        : n/a")
    print(f"  Best so far     : {s.get('best_loss',   float('nan')):.6f}")
    print()
    print(f"  Elapsed : {fmt_time(s.get('elapsed_s', 0))}")
    print(f"  ETA     : {s.get('eta', 'unknown')}")
    print()

    age_str = f"{int(age)}s ago"
    if age > STALE_WARN:
        print(f"  WARNING: last update was {age_str} -- job may have stalled or finished.")
    else:
        print(f"  Last update : {updated.strftime('%Y-%m-%d %H:%M:%S')}  ({age_str})")

    # ---- tail of log file ----
    if log_file.exists():
        lines = log_file.read_text().splitlines()
        tail  = lines[-8:] if len(lines) >= 8 else lines
        print()
        print("  --- last log lines ---")
        for ln in tail:
            print(f"  {ln}")

    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
