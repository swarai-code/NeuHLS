"""
run_all.py — End-to-end pipeline orchestrator for test_svhn.

Steps:
  1. Train baseline MLP            (train_baseline.py)
  2. Extract activation traces      (extract_traces.py)
  3. Run PySR per experiment        (run_pysr.py)
  4. Select best equations          (select_equation.py)
  5. Evaluate hybrid models         (eval_hybrid.py)
  6. Generate HLS C++ code          (generate_hls.py)

Use --smoke-test to verify the whole pipeline quickly with tiny data.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config

PYTHON = sys.executable


def _run(cmd, step, required=False):
    print(f"\n{'='*60}")
    print(f" STEP: {step}")
    print(f" CMD : {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    ok = result.returncode == 0
    if not ok:
        msg = f"[run_all] Step '{step}' failed (exit code {result.returncode})"
        if required:
            print(f"\n{msg} — aborting pipeline.")
            sys.exit(result.returncode)
        print(f"\n[run_all] WARNING: {msg}")
    return ok


def main(args):
    script_dir = Path(__file__).parent

    smoke   = ["--smoke-test"] if args.smoke_test else []
    device  = ["--device", args.device]

    experiments = args.experiments or config.EXPERIMENTS
    exp_flags_flat = experiments   # list

    # ── 1. Train baseline ─────────────────────────────────────────────────────
    if not args.skip_train:
        epoch_flag = ["--epochs", str(args.epochs)]
        _run(
            [PYTHON, str(script_dir / "train_baseline.py")] + epoch_flag + smoke + device,
            "Train Baseline MLP",
            required=True,
        )
    else:
        print("[run_all] Skipping baseline training (--skip-train)")

    # ── 2. Extract traces ─────────────────────────────────────────────────────
    trace_args = []
    if args.max_train_samples:
        trace_args += ["--max-train-samples", str(args.max_train_samples)]
    if args.max_test_samples:
        trace_args += ["--max-test-samples", str(args.max_test_samples)]
    _run(
        [PYTHON, str(script_dir / "extract_traces.py")] + trace_args + smoke,
        "Extract Activation Traces",
        required=True,
    )

    # ── 3. Run PySR ───────────────────────────────────────────────────────────
    if args.run_pysr and not args.skip_pysr:
        pysr_args = ["--niterations", str(args.pysr_iterations)] if args.pysr_iterations else []
        for exp in exp_flags_flat:
            _run(
                [PYTHON, str(script_dir / "run_pysr.py"),
                 "--experiment", exp] + pysr_args + smoke,
                f"PySR  {exp}",
            )
    elif args.skip_pysr:
        print("[run_all] Skipping PySR (--skip-pysr)")
    else:
        print("[run_all] Skipping PySR (pass --run-pysr to enable)")

    # ── 4. Select equations ───────────────────────────────────────────────────
    _run(
        [PYTHON, str(script_dir / "select_equation.py"), "--all"],
        "Select Best Equations",
    )

    # ── 5. Evaluate hybrid models ─────────────────────────────────────────────
    _run(
        [PYTHON, str(script_dir / "eval_hybrid.py"), "--all"] + smoke + device,
        "Evaluate Hybrid Models",
    )

    # ── 6. Fine-tune kept layers after symbolic replacement ───────────────────
    ft_args = ["--epochs", str(args.finetune_epochs), "--lr", str(args.finetune_lr)]
    _run(
        [PYTHON, str(script_dir / "finetune_hybrid.py"), "--all"] + ft_args + smoke + device,
        "Fine-tune Hybrid Models",
    )

    # ── 7. Generate HLS code ──────────────────────────────────────────────────
    _run(
        [PYTHON, str(script_dir / "generate_hls.py"), "--all"],
        "Generate HLS C++ Code",
    )

    print(f"\n{'='*60}")
    print(" Pipeline complete!")
    print(f" Results: {config.RESULTS_DIR / 'final_results.csv'}")
    print(f"{'='*60}\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full NeuSym-HLS pipeline on SVHN (1 vs 7)"
    )
    p.add_argument("--smoke-test",         action="store_true",
                   help="Run with tiny dataset/iterations for quick verification")
    p.add_argument("--skip-train",         action="store_true",
                   help="Skip baseline training (assumes checkpoint exists)")
    p.add_argument("--skip-pysr",          action="store_true",
                   help="Skip PySR (assumes hall-of-fame CSVs exist)")
    p.add_argument("--run-pysr",           action="store_true",
                   help="Enable PySR step (disabled by default to avoid long runs)")
    p.add_argument("--experiments",        nargs="+",
                   choices=config.EXPERIMENTS, default=None,
                   help="Which experiments to include (default: all)")
    p.add_argument("--max-train-samples",  type=int, default=None)
    p.add_argument("--max-test-samples",   type=int, default=None)
    p.add_argument("--epochs",             type=int,   default=config.NUM_EPOCHS)
    p.add_argument("--pysr-iterations",    type=int,   default=None,
                   help="Override PySR niterations")
    p.add_argument("--finetune-epochs",    type=int,   default=10,
                   help="Epochs for fine-tuning kept layers (default: 10)")
    p.add_argument("--finetune-lr",        type=float, default=1e-4,
                   help="Learning rate for fine-tuning (default: 1e-4)")
    p.add_argument("--device",             type=str,   default=config.DEVICE)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
