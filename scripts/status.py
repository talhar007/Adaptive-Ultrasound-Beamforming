"""Training status monitor for ABLE++.

Run this at any time from any login node to see the current state of your
running (or most-recently-finished) training job:

    python status.py                       # one-shot snapshot
    watch -n 30 python status.py           # refresh every 30 seconds live

Reads  Trained_Checkpoints/checkpoints/status.json  (written by run_training.py
every --log_interval steps) and  Trained_Checkpoints/checkpoints/training.log
(last few lines).
"""
import json
import sys
from datetime import datetime
from pathlib import Path


STATUS_FILE = Path('Trained_Checkpoints/checkpoints/status.json')
LOG_FILE    = Path('Trained_Checkpoints/checkpoints/training.log')
STALE_WARN  = 120   # seconds before we warn the job may have died


def bar(current, total, width=40):
    filled = int(width * current / max(total, 1))
    return '[' + '#' * filled + '-' * (width - filled) + ']'


def fmt_time(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def main():
    # ---- status.json ----
    if not STATUS_FILE.exists():
        print(f"No status file found at {STATUS_FILE}")
        print("Either training hasn't started yet, or the checkpoint_dir differs.")
        sys.exit(0)

    with open(STATUS_FILE) as f:
        s = json.load(f)

    updated = datetime.fromisoformat(s['updated'])
    age     = (datetime.now() - updated).total_seconds()

    stage    = s.get('stage', '?')
    s_step   = s.get('stage_step', 0)
    s_total  = s.get('stage_total', 1)
    pct      = 100 * s_step / max(s_total, 1)
    progress = bar(s_step, s_total)

    print()
    print("=" * 60)
    print("  ABLE++ Training Status")
    print("=" * 60)
    print(f"  Stage   : {stage.upper()}")
    print(f"  Progress: {progress} {pct:5.1f}%  ({s_step}/{s_total} steps)")
    print()
    print(f"  Loss (total)  : {s.get('loss',        float('nan')):.6f}")
    print(f"  Loss (image)  : {s.get('image_loss',  float('nan')):.6f}")
    print(f"  Loss (unity)  : {s.get('unity_loss',  float('nan')):.6f}")
    print(f"  Best so far   : {s.get('best_loss',   float('nan')):.6f}")
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
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()
        tail  = lines[-8:] if len(lines) >= 8 else lines
        print()
        print("  --- last log lines ---")
        for ln in tail:
            print(f"  {ln}")

    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
