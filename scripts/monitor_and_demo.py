#!/usr/bin/env python3
"""Monitor training progress and auto-run demo when complete.

Run this in a terminal:
    python monitor_and_demo.py

It will:
  1. Monitor training progress in real time
  2. Show progress bar and ETA
  3. When training finishes, automatically run the full demo
  4. Generate visualizations for your professor
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def bar(current, total, width=50):
    """Progress bar."""
    if total == 0:
        return "[" + "-" * width + "]"
    filled = int(width * current / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def monitor_training():
    """Monitor status.json and print progress."""
    status_file = Path("Trained_Checkpoints/checkpoints/status.json")
    training_log = Path("Trained_Checkpoints/checkpoints/training.log")

    print("\n" + "=" * 70)
    print("  ABLE++ Training Monitor")
    print("=" * 70 + "\n")

    last_step = 0
    training_done = False

    while not training_done:
        if status_file.exists():
            try:
                with open(status_file) as f:
                    s = json.load(f)

                step = s.get("stage_step", 0)
                total = s.get("stage_total", 5000)
                loss = s.get("loss", 0)
                best_loss = s.get("best_loss", float("inf"))
                eta = s.get("eta", "?")
                elapsed = s.get("elapsed_s", 0)

                pct = 100 * step / max(total, 1)

                # Print update every 100 steps or when status changes
                if step != last_step and (step % 100 == 0 or step == total):
                    progress = bar(step, total)
                    print(
                        f"\r  {progress} {pct:5.1f}% | "
                        f"Step {step:5d}/{total} | "
                        f"Loss: {loss:8.4f} | "
                        f"Best: {best_loss:8.4f} | "
                        f"ETA: {eta:8s}",
                        end="",
                        flush=True
                    )
                    last_step = step

                if step >= total and total > 0:
                    training_done = True
                    print()  # newline
                    print(
                        f"\n✓ Training completed at {datetime.now().strftime('%H:%M:%S')}"
                    )
                    print(f"  Final loss: {loss:.4f}")
                    print(f"  Best loss:  {best_loss:.4f}")
                    print(f"  Total time: {int(elapsed)}s\n")
                    break
            except json.JSONDecodeError:
                pass

        time.sleep(5)


def run_demo():
    """Run the full demo pipeline."""
    print("=" * 70)
    print("  Running Full Demo Pipeline")
    print("=" * 70 + "\n")

    cmd = [
        sys.executable,
        "demo_results.py",
        "--M", "64",
        "--nx", "128",
        "--nz", "128",
        "--n_test", "3",
        "--output_dir", "Results/demo_output",
    ]

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path.cwd())

    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("  Demo Complete!")
        print("=" * 70)
        print(f"\nResults saved to:")
        print(f"  - Results/demo_output/comparison_results.txt      (metrics)")
        print(f"  - Results/demo_output/case_XX_reconstruction.png  (B-mode images)")
        print(f"  - Results/demo_output/case_XX_tensors.pt          (raw data)\n")
        print("Ready to present to your professor!\n")
    else:
        print(f"\nERROR: Demo failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    print("\nStarting training monitor...\n")

    # Monitor until complete
    monitor_training()

    # Run demo
    run_demo()


if __name__ == "__main__":
    main()
