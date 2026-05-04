"""
eval_hybrid.py — Evaluate baseline and all six symbolic hybrid models.

Outputs:
  outputs/results/final_results.csv
  outputs/results/final_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import config
from data_utils import get_loaders
from models import BaselineMLP, load_baseline, build_hybrid
from op_counter import count_ops


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _predict(model: nn.Module, loader, device: str):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels = [], []
    total_loss = 0.0
    total_n    = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit  = model(xb).squeeze(1)
            loss   = criterion(logit, yb)
            total_loss += loss.item() * xb.size(0)
            total_n    += xb.size(0)
            all_logits.append(logit.cpu())
            all_labels.append(yb.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds  = (torch.sigmoid(logits) >= 0.5).long()
    bce    = total_loss / total_n
    acc    = (preds == labels.long()).float().mean().item()
    return logits.numpy(), labels.numpy(), preds.numpy(), bce, acc


def _latency_ms(model: nn.Module, loader, device: str, n_repeats: int = 20) -> float:
    """Average per-sample inference latency in milliseconds."""
    model.eval()
    model.to(device)
    xb, _ = next(iter(loader))
    xb = xb.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            model(xb)

    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model(xb)
            times.append((time.perf_counter() - t0) / xb.size(0) * 1000)

    return float(np.mean(times))


def _metrics(labels, preds, bce) -> dict:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy":         float(np.mean(labels == preds)),
        "loss":             float(bce),
        "precision":        float(prec),
        "recall":           float(rec),
        "f1":               float(f1),
        "confusion_matrix": cm.tolist(),
    }


def _cost_breakdown(layer: str, op_info: dict) -> dict:
    sym_cost   = op_info.get("weighted_cost", 0)
    kept_macs  = config.KEPT_MACS[layer]
    rep_macs   = config.REPLACED_MACS[layer]
    total_est  = kept_macs * 2 + sym_cost   # kept as MACs→FLOPs, symbolic as ops

    return {
        "baseline_macs_total":    config.BASELINE_MACS,
        "baseline_flops_total":   config.BASELINE_FLOPS,
        "baseline_macs_kept":     kept_macs,
        "replaced_macs":          rep_macs,
        "symbolic_operator_count": op_info.get("total_ops", 0),
        "symbolic_cost":          sym_cost,
        "estimated_total_cost":   total_est,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_baseline(loader, device: str) -> dict:
    ckpt = config.CKPT_DIR / "baseline.pt"
    if not ckpt.exists():
        print("[eval] Baseline checkpoint not found — skipping.")
        return {}

    model = load_baseline(ckpt, device=device)
    _, labels, preds, bce, acc = _predict(model, loader, device)
    lat_cpu = _latency_ms(model, loader, "cpu")

    row = {
        "model_id":    "Baseline-MLP",
        "replacement": "none",
        "operator_set": "none",
        **_metrics(labels, preds, bce),
        "baseline_macs_total":     config.BASELINE_MACS,
        "baseline_flops_total":    config.BASELINE_FLOPS,
        "baseline_macs_kept":      config.BASELINE_MACS,
        "replaced_macs":           0,
        "symbolic_operator_count": 0,
        "symbolic_cost":           0,
        "estimated_total_cost":    config.BASELINE_FLOPS,
        "cpu_latency_ms":          round(lat_cpu, 4),
        "gpu_latency_ms":          None,
        "equation_complexity":     0,
        "equation":                "none",
    }
    print(f"[eval] Baseline  acc={acc:.4f}  loss={bce:.4f}")
    return row


def eval_hybrid_experiment(exp_name: str, loader, device: str) -> dict | None:
    layer = config.get_layer(exp_name)
    opset = config.get_opset(exp_name)

    eq_path = config.EQ_DIR / f"{exp_name}_best_equation.json"
    if not eq_path.exists():
        print(f"[eval] {exp_name}: equation file not found ({eq_path}) — skipping.")
        return None

    with open(eq_path) as f:
        eq_data = json.load(f)

    equation_str = eq_data.get("equation", "")
    complexity   = eq_data.get("complexity", 0)

    ckpt = config.CKPT_DIR / "baseline.pt"
    if not ckpt.exists():
        print(f"[eval] {exp_name}: baseline checkpoint not found — skipping.")
        return None

    baseline = load_baseline(ckpt, device=device)
    model    = build_hybrid(baseline, equation_str, layer)
    model.eval()

    try:
        _, labels, preds, bce, acc = _predict(model, loader, device)
    except Exception as e:
        print(f"[eval] {exp_name}: inference error — {e}")
        return None

    lat_cpu = _latency_ms(model, loader, "cpu")
    lat_gpu = None
    if torch.cuda.is_available() and device == "cuda":
        lat_gpu = _latency_ms(model, loader, "cuda")

    op_info = {}
    try:
        op_info = count_ops(equation_str)
    except Exception as e:
        print(f"[eval] {exp_name}: op_counter error — {e}")

    row = {
        "model_id":     exp_name,
        "replacement":  layer,
        "operator_set": opset,
        **_metrics(labels, preds, bce),
        **_cost_breakdown(layer, op_info),
        "cpu_latency_ms":      round(lat_cpu, 4),
        "gpu_latency_ms":      round(lat_gpu, 4) if lat_gpu else None,
        "equation_complexity": complexity,
        "equation":            equation_str,
    }
    print(f"[eval] {exp_name:<18s}  acc={acc:.4f}  loss={bce:.4f}  "
          f"sym_cost={op_info.get('weighted_cost', '?')}")
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = args.device
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, test_loader = get_loaders(smoke_test=args.smoke_test)

    results = []

    # Always evaluate baseline
    row = eval_baseline(test_loader, device)
    if row:
        results.append(row)

    experiments = config.EXPERIMENTS if args.all else ([args.experiment] if args.experiment else [])
    for exp in experiments:
        row = eval_hybrid_experiment(exp, test_loader, device)
        if row:
            results.append(row)

    if not results:
        print("[eval] No results to save.")
        return

    df = pd.DataFrame(results)
    csv_path  = config.RESULTS_DIR / "final_results.csv"
    json_path = config.RESULTS_DIR / "final_results.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved:")
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    print("\n" + df[["model_id", "accuracy", "loss", "f1",
                      "symbolic_cost", "estimated_total_cost"]].to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate baseline + symbolic hybrid models")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all",        action="store_true")
    grp.add_argument("--experiment", type=str, choices=config.EXPERIMENTS)
    p.add_argument("--checkpoint",   type=str,  default=str(config.CKPT_DIR / "baseline.pt"))
    p.add_argument("--device",       type=str,  default=config.DEVICE)
    p.add_argument("--smoke-test",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
