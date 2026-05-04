"""
run_pysr.py — Run PySR symbolic regression for each experiment variant.

Experiments:
  SR-1L-POL   SR-1L-SRL   SR-1L-SCE
  SR-2L-POL   SR-2L-SRL   SR-2L-SCE

Outputs per experiment saved under:
  outputs/pysr/{experiment_name}/
    hall_of_fame.csv
    run_config.json
    run_log.txt
    model.pkl          (if serialisable)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config

# ── PySR availability check ────────────────────────────────────────────────────
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False


def _check_pysr():
    if not PYSR_AVAILABLE:
        print(
            "\n[ERROR] PySR is not installed.\n"
            "Install with:  pip install pysr\n"
            "PySR requires Julia. First run (as admin if needed):\n"
            "  python -c \"import juliapkg; juliapkg.resolve()\"\n"
            "or follow: https://astroautomata.com/PySR/\n"
        )
        sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_traces(layer: str, max_samples: int = None):
    """Load input/output CSVs for the given layer ('1L' or '2L')."""
    prefix = "fc3" if layer == "1L" else "2L"
    x_path = config.TRACE_DIR / f"pysr_input_{prefix}_train.csv"
    y_path = config.TRACE_DIR / f"pysr_output_{prefix}_train.csv"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Trace files not found for layer {layer}. "
            f"Run extract_traces.py first.\n  {x_path}\n  {y_path}"
        )

    X = pd.read_csv(x_path).values.astype(np.float32)
    y = pd.read_csv(y_path).values.ravel().astype(np.float32)

    if max_samples and max_samples < X.shape[0]:
        rng = np.random.default_rng(config.SEED)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    # Sanitise
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    print(f"  Trace loaded: X={X.shape}  y={y.shape}  "
          f"y_mean={y.mean():.4f}  y_std={y.std():.4f}")
    return X, y


def _build_model(opset_name: str, niterations: int, populations: int,
                 maxsize: int, timeout) -> "PySRRegressor":
    ops = config.OPERATOR_SETS[opset_name]

    import sympy as sp

    # Tell PySR how to convert custom operators back to SymPy expressions
    extra_sympy_mappings = {}
    if any("square" in op for op in ops["unary_operators"]):
        extra_sympy_mappings["square"] = lambda x: x ** 2
    if any("relu" in op for op in ops["unary_operators"]):
        extra_sympy_mappings["relu"] = lambda x: sp.Piecewise((x, x > 0), (sp.Integer(0), True))

    # Operator names present in this experiment's unary set
    # (Julia-style defs like "square(x::T) where {T} = x*x" need splitting on "(")
    unary_names = {op.split("(")[0].strip() for op in ops["unary_operators"]}

    kwargs = dict(
        niterations      = niterations,
        populations      = populations,
        maxsize          = maxsize,
        binary_operators = ops["binary_operators"],
        unary_operators  = ops["unary_operators"],
        extra_sympy_mappings = extra_sympy_mappings,
        random_state     = config.PYSR_SEED,
        verbosity        = 1,
        progress         = True,
    )

    # Constrain argument complexity for transcendental ops — but only when they
    # are actually in the operator set. Passing a constraint for an absent op
    # causes PySR to raise ValueError.
    transcendental_constraints = {
        op: (-5, 5) for op in ["exp", "sin", "cos"] if op in unary_names
    }
    if transcendental_constraints:
        kwargs["constraints"] = transcendental_constraints

    if timeout:
        kwargs["timeout_in_seconds"] = timeout * 60

    return PySRRegressor(**kwargs)


def run_experiment(exp_name: str, args):
    _check_pysr()

    layer    = config.get_layer(exp_name)
    opset    = config.get_opset(exp_name)
    out_dir  = config.PYSR_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"=== PySR run: {exp_name}  layer={layer}  opset={opset} ===")

    # Use local variables so smoke_test overrides don't mutate the shared args
    # namespace across multiple experiments when --all is used.
    max_samples = args.max_trace_samples
    niterations = args.niterations
    populations = args.populations
    maxsize     = args.maxsize

    if args.smoke_test:
        max_samples = 200
        niterations = 5
        populations = 4
        maxsize     = 10

    logger.info("Loading traces …")
    try:
        X, y = _load_traces(layer, max_samples=max_samples)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Save run config
    run_cfg = {
        "experiment":     exp_name,
        "layer":          layer,
        "opset":          opset,
        "n_train":        int(X.shape[0]),
        "n_features":     int(X.shape[1]),
        "niterations":    niterations,
        "populations":    populations,
        "maxsize":        maxsize,
        "timeout_mins":   args.timeout,
        "smoke_test":     args.smoke_test,
        "max_samples":    max_samples,
        "operator_set":   config.OPERATOR_SETS[opset],
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    logger.info(f"Building PySR model (niterations={niterations}) …")
    model = _build_model(opset, niterations, populations, maxsize, args.timeout)

    # PySR writes its own hall_of_fame file; point it to our directory
    hof_path = str(out_dir / "hall_of_fame.csv")

    t0 = time.time()
    try:
        model.fit(X, y, variable_names=[f"x{i}" for i in range(X.shape[1])])
    except Exception:
        logger.error("PySR fit failed:\n" + traceback.format_exc())
        return
    elapsed = time.time() - t0
    logger.info(f"PySR fit done in {elapsed:.1f}s")

    # ── Save outputs ───────────────────────────────────────────────────────────
    # Hall of fame (equations for all complexities found)
    try:
        hof: pd.DataFrame = model.equations_
        if hof is not None and len(hof) > 0:
            hof.to_csv(hof_path, index=False)
            logger.info(f"Hall of fame saved ({len(hof)} equations) → {hof_path}")
        else:
            logger.warning("Hall of fame is empty.")
    except Exception as e:
        logger.warning(f"Could not save hall of fame: {e}")

    # Best equation
    try:
        best_eq = model.sympy()
        logger.info(f"Best equation: {best_eq}")
        with open(out_dir / "best_equation.txt", "w") as f:
            f.write(str(best_eq) + "\n")
    except Exception as e:
        logger.warning(f"Could not get best equation: {e}")

    # Pickle model
    try:
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("Model pickled.")
    except Exception as e:
        logger.warning(f"Could not pickle model: {e}")

    logger.info(f"=== {exp_name} complete ===\n")


def parse_args():
    p = argparse.ArgumentParser(description="Run PySR symbolic regression")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--experiment",  type=str, choices=config.EXPERIMENTS,
                     help="Single experiment to run")
    grp.add_argument("--all",         action="store_true",
                     help="Run all experiments sequentially")
    p.add_argument("--niterations",       type=int,   default=config.PYSR_NITERATIONS)
    p.add_argument("--populations",       type=int,   default=config.PYSR_POPULATIONS)
    p.add_argument("--maxsize",           type=int,   default=config.PYSR_MAXSIZE)
    p.add_argument("--max-trace-samples", type=int,   default=None,
                   help="Limit trace rows fed to PySR")
    p.add_argument("--timeout",           type=float, default=None,
                   help="Timeout per experiment in minutes")
    p.add_argument("--smoke-test",        action="store_true",
                   help="Tiny run for pipeline verification")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiments = config.EXPERIMENTS if args.all else [args.experiment]
    for exp in experiments:
        run_experiment(exp, args)
