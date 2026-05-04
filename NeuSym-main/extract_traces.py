"""
extract_traces.py — Extract activation traces from the trained baseline for PySR.

Traces saved:
  1L  (fc3 inputs):  h2 [N, 128]  →  logit [N, 1]
  2L  (2L inputs):   h1 [N, 512]  →  logit [N, 1]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from data_utils import get_tensors
from models import BaselineMLP, load_baseline


def _sanitize(x_arr: np.ndarray, y_arr: np.ndarray):
    """Remove rows containing NaN or Inf from x or y."""
    mask = np.isfinite(x_arr).all(axis=1) & np.isfinite(y_arr).ravel()
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"  [sanitize] removed {n_removed} rows with NaN/Inf")
    return x_arr[mask], y_arr[mask]


@torch.no_grad()
def _extract_activations(model: BaselineMLP, x: torch.Tensor, batch_size: int = 1024):
    """Run forward pass and collect h1, h2, logit for the full tensor x."""
    model.eval()
    h1_list, h2_list, logit_list = [], [], []

    for i in range(0, x.size(0), batch_size):
        xb = x[i : i + batch_size]
        logit, h1, h2 = model(xb, return_activations=True)
        h1_list.append(h1.cpu().numpy())
        h2_list.append(h2.cpu().numpy())
        logit_list.append(logit.cpu().numpy())

    return (
        np.concatenate(h1_list, axis=0),       # [N, 512]
        np.concatenate(h2_list, axis=0),        # [N, 128]
        np.concatenate(logit_list, axis=0),     # [N, 1]
    )


def _save_trace(x_arr, y_arr, prefix: str, tag: str):
    """
    Save (x, y) as CSVs with column names x0, x1, ... and y.

    prefix: e.g. 'fc3' or '2L'
    tag:    'train' or 'test'
    """
    x_cols = [f"x{i}" for i in range(x_arr.shape[1])]
    x_df   = pd.DataFrame(x_arr, columns=x_cols)
    y_df   = pd.DataFrame(y_arr.ravel(), columns=["y"])

    in_path  = config.TRACE_DIR / f"pysr_input_{prefix}_{tag}.csv"
    out_path = config.TRACE_DIR / f"pysr_output_{prefix}_{tag}.csv"

    x_df.to_csv(in_path,  index=False)
    y_df.to_csv(out_path, index=False)

    print(f"  [{tag:5s}] {prefix} input  → {in_path}  shape={x_arr.shape}")
    print(f"  [{tag:5s}] {prefix} output → {out_path} shape={y_arr.shape}")

    return {
        "input_path":  str(in_path),
        "output_path": str(out_path),
        "shape_input":  list(x_arr.shape),
        "shape_output": list(y_arr.shape),
        "x_mean":  float(x_arr.mean()),
        "x_std":   float(x_arr.std()),
        "y_mean":  float(y_arr.mean()),
        "y_std":   float(y_arr.std()),
        "y_min":   float(y_arr.min()),
        "y_max":   float(y_arr.max()),
    }


def extract(args):
    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    x_tr, _, x_te, _ = get_tensors(
        smoke_test=args.smoke_test,
        max_train=args.max_train_samples,
        max_test=args.max_test_samples,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading baseline from {ckpt_path} …")
    model = load_baseline(ckpt_path, device="cpu")

    print("Extracting training activations …")
    t0 = time.time()
    h1_tr, h2_tr, logit_tr = _extract_activations(model, x_tr)
    print(f"  done in {time.time()-t0:.1f}s")

    print("Extracting test activations …")
    t0 = time.time()
    h1_te, h2_te, logit_te = _extract_activations(model, x_te)
    print(f"  done in {time.time()-t0:.1f}s")

    summary = {}

    # Compute one consistent mask across h1, h2, and logit so that 1L and 2L
    # traces correspond to exactly the same sample rows.
    print("\nSanitising traces …")
    mask_tr = (np.isfinite(h1_tr).all(axis=1) &
               np.isfinite(h2_tr).all(axis=1) &
               np.isfinite(logit_tr).ravel())
    mask_te = (np.isfinite(h1_te).all(axis=1) &
               np.isfinite(h2_te).all(axis=1) &
               np.isfinite(logit_te).ravel())

    if (~mask_tr).sum() > 0:
        print(f"  [sanitize] removed {(~mask_tr).sum()} train rows with NaN/Inf")
    if (~mask_te).sum() > 0:
        print(f"  [sanitize] removed {(~mask_te).sum()} test rows with NaN/Inf")

    h1_tr, h2_tr, logit_tr = h1_tr[mask_tr], h2_tr[mask_tr], logit_tr[mask_tr]
    h1_te, h2_te, logit_te = h1_te[mask_te], h2_te[mask_te], logit_te[mask_te]

    # ── 1L traces  (h2 → logit) ───────────────────────────────────────────────
    print("\nSaving 1L traces (h2 → logit) …")
    summary["1L_train"] = _save_trace(h2_tr, logit_tr, "fc3", "train")
    summary["1L_test"]  = _save_trace(h2_te, logit_te, "fc3", "test")

    # ── 2L traces  (h1 → logit) ───────────────────────────────────────────────
    print("\nSaving 2L traces (h1 → logit) …")
    summary["2L_train"] = _save_trace(h1_tr, logit_tr, "2L", "train")
    summary["2L_test"]  = _save_trace(h1_te, logit_te, "2L", "test")

    summary_path = config.TRACE_DIR / "trace_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Extract activation traces from baseline MLP")
    p.add_argument("--checkpoint",        type=str,  default=str(config.CKPT_DIR / "baseline.pt"))
    p.add_argument("--max-train-samples", type=int,  default=None,
                   help="Cap training trace size (PySR can be slow on large inputs)")
    p.add_argument("--max-test-samples",  type=int,  default=None)
    p.add_argument("--smoke-test",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    extract(parse_args())
