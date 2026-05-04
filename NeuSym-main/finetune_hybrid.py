"""
finetune_hybrid.py — Fine-tune kept neural layers of each hybrid model.

After PySR replaces the tail layer(s) with a fixed symbolic expression, fc1
and fc2 are no longer optimal — they were trained to feed a learned linear
head, not a fixed math formula.  Fine-tuning lets them adapt to the symbolic
head while keeping the expression itself frozen.

Procedure per experiment:
  1. Load the symbolic expression from outputs/equations/
  2. Deep-copy the baseline so the original weights are not modified
  3. Build the hybrid on the copy
  4. Fine-tune fc1 (+ fc2 for 1L) end-to-end with BCEWithLogitsLoss on real labels
  5. Evaluate and save fine-tuned checkpoint + results

Outputs:
  outputs/checkpoints/{exp}_finetuned.pt
  outputs/results/finetuned_results.csv
  outputs/results/finetuned_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import pandas as pd
import torch
import torch.nn as nn

import config
from data_utils import get_loaders
from eval_hybrid import _cost_breakdown, _latency_ms, _metrics, _predict
from models import build_hybrid, load_baseline
from op_counter import count_ops


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning loop
# ──────────────────────────────────────────────────────────────────────────────

def _finetune(model: nn.Module, train_loader, test_loader,
              device: str, epochs: int, lr: float) -> list:
    """Fine-tune on real labels. Returns per-epoch log."""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logit = model(xb).squeeze(1)
            loss  = criterion(logit, yb)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * xb.size(0)
            total_n       += xb.size(0)
            with torch.no_grad():
                preds = (torch.sigmoid(logit) >= 0.5).long()
                total_correct += (preds == yb.long()).sum().item()

        train_loss = total_loss / total_n
        train_acc  = total_correct / total_n
        _, _, _, bce, test_acc = _predict(model, test_loader, device)

        log.append(dict(
            epoch      = epoch,
            train_loss = round(train_loss, 6),
            train_acc  = round(train_acc, 6),
            test_loss  = round(bce, 6),
            test_acc   = round(test_acc, 6),
        ))
        print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    return log


# ──────────────────────────────────────────────────────────────────────────────
# Per-experiment entry point
# ──────────────────────────────────────────────────────────────────────────────

def finetune_experiment(exp_name: str, train_loader, test_loader,
                        device: str, epochs: int, lr: float) -> dict | None:
    layer = config.get_layer(exp_name)
    opset = config.get_opset(exp_name)

    eq_path = config.EQ_DIR / f"{exp_name}_best_equation.json"
    if not eq_path.exists():
        print(f"[finetune] {exp_name}: equation file not found — skipping.")
        return None

    with open(eq_path) as f:
        eq_data = json.load(f)
    equation_str = eq_data.get("equation", "")
    complexity   = eq_data.get("complexity", 0)

    ckpt = config.CKPT_DIR / "baseline.pt"
    if not ckpt.exists():
        print(f"[finetune] baseline checkpoint not found — skipping.")
        return None

    baseline = load_baseline(ckpt, device=device)

    # Deep copy so fine-tuning does not modify the shared baseline weights
    model = build_hybrid(copy.deepcopy(baseline), equation_str, layer)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[finetune] {exp_name}  layer={layer}  "
          f"epochs={epochs}  lr={lr}  trainable_params={n_params:,}")

    t0  = time.time()
    log = _finetune(model, train_loader, test_loader, device, epochs, lr)
    elapsed = time.time() - t0

    # Save fine-tuned checkpoint
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_out = config.CKPT_DIR / f"{exp_name}_finetuned.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "experiment":       exp_name,
        "epochs":           epochs,
        "lr":               lr,
        "final_test_acc":   log[-1]["test_acc"],
    }, ckpt_out)

    # Final evaluation
    _, labels, preds, bce, acc = _predict(model, test_loader, device)
    lat_cpu = _latency_ms(model, test_loader, "cpu")

    op_info = {}
    try:
        op_info = count_ops(equation_str)
    except Exception as e:
        print(f"[finetune] {exp_name}: op_counter error — {e}")

    row = {
        "model_id":            f"{exp_name}-FT",
        "base_experiment":     exp_name,
        "replacement":         layer,
        "operator_set":        opset,
        **_metrics(labels, preds, bce),
        **_cost_breakdown(layer, op_info),
        "cpu_latency_ms":      round(lat_cpu, 4),
        "equation_complexity": complexity,
        "equation":            equation_str,
        "finetune_epochs":     epochs,
        "finetune_lr":         lr,
        "elapsed_sec":         round(elapsed, 2),
    }
    print(f"[finetune] {exp_name}-FT  acc={acc:.4f}  loss={bce:.4f}")
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_finetune(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[finetune] CUDA requested but unavailable — falling back to CPU.")
        args.device = "cpu"
    device = args.device

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = get_loaders(smoke_test=args.smoke_test)

    experiments = config.EXPERIMENTS if args.all else [args.experiment]
    results = []

    for exp in experiments:
        row = finetune_experiment(
            exp, train_loader, test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )
        if row:
            results.append(row)

    if not results:
        print("[finetune] No results to save.")
        return

    df = pd.DataFrame(results)
    csv_path  = config.RESULTS_DIR / "finetuned_results.csv"
    json_path = config.RESULTS_DIR / "finetuned_results.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFine-tuned results saved:")
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    print("\n" + df[["model_id", "accuracy", "f1",
                      "symbolic_cost", "estimated_total_cost"]].to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune kept neural layers after symbolic replacement"
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all",        action="store_true")
    grp.add_argument("--experiment", type=str, choices=config.EXPERIMENTS)
    p.add_argument("--epochs",     type=int,   default=10,
                   help="Fine-tuning epochs (default: 10; lower than baseline "
                        "to avoid destroying already-trained layers)")
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Learning rate (default: 1e-4, 10x lower than baseline)")
    p.add_argument("--device",     type=str,   default=config.DEVICE)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_finetune(parse_args())
