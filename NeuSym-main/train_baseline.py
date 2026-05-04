"""
train_baseline.py — Train the BaselineMLP on SVHN (digits 1 vs 7).
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from data_utils import get_loaders
from models import BaselineMLP


def set_seeds(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit = model(xb).squeeze(1)
            loss  = criterion(logit, yb)
            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logit) >= 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total   += xb.size(0)
    return total_loss / total, correct / total


def train(args):
    set_seeds()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable — falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
    )

    model     = BaselineMLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = config.RESULTS_DIR / "baseline_training_log.csv"
    log_rows  = []

    best_test_acc = 0.0
    best_epoch    = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logit = model(xb).squeeze(1)
            loss  = criterion(logit, yb)
            loss.backward()
            optimizer.step()
            train_loss  += loss.item() * xb.size(0)
            train_total += xb.size(0)
            with torch.no_grad():
                preds = (torch.sigmoid(logit) >= 0.5).long()
                train_correct += (preds == yb.long()).sum().item()

        train_loss /= train_total
        train_acc   = train_correct / train_total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        row = dict(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            train_acc=round(train_acc, 6),
            test_loss=round(test_loss, 6),
            test_acc=round(test_acc, 6),
        )
        log_rows.append(row)
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch    = epoch
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": epoch,
                 "test_acc": test_acc,
                 "test_loss": test_loss},
                config.CKPT_DIR / "baseline.pt",
            )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")

    # Save training log CSV
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        writer.writeheader()
        writer.writerows(log_rows)

    # Final evaluation on best checkpoint
    best_state = torch.load(config.CKPT_DIR / "baseline.pt",
                            map_location=device, weights_only=True)
    model.load_state_dict(best_state["model_state_dict"])
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)

    metrics = {
        "model": "BaselineMLP",
        "best_epoch": best_epoch,
        "final_test_accuracy": round(final_acc, 6),
        "final_test_loss": round(final_loss, 6),
        "train_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "smoke_test": args.smoke_test,
        "elapsed_sec": round(elapsed, 2),
        "baseline_macs_total": config.BASELINE_MACS,
        "baseline_flops_total": config.BASELINE_FLOPS,
    }

    with open(config.RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFinal baseline accuracy: {final_acc:.4f}")
    print(f"Checkpoint: {config.CKPT_DIR / 'baseline.pt'}")
    print(f"Metrics:    {config.RESULTS_DIR / 'baseline_metrics.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline MLP on SVHN (1 vs 7)")
    p.add_argument("--epochs",      type=int,   default=config.NUM_EPOCHS)
    p.add_argument("--batch-size",  type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    p.add_argument("--device",      type=str,   default=config.DEVICE)
    p.add_argument("--smoke-test",  action="store_true",
                   help="Use tiny dataset subset for quick pipeline verification")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
