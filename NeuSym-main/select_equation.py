"""
select_equation.py — Choose the best equation from a PySR hall-of-fame CSV.

Selection policy:
  1. Keep only rows where complexity <= complexity_budget.
  2. Among those, prefer minimum loss.
  3. If multiple rows share the same loss, prefer lower symbolic operator cost.

Outputs:
  outputs/equations/{exp}_best_equation.txt
  outputs/equations/{exp}_best_equation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import config
from op_counter import count_ops


# Columns PySR might use for the equation string and loss
_EQ_COLS   = ["equation", "sympy_format", "lambda_format"]
_LOSS_COLS = ["loss", "score"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_hof(exp_name: str) -> pd.DataFrame | None:
    path = config.PYSR_DIR / exp_name / "hall_of_fame.csv"
    if not path.exists():
        print(f"[select] Hall-of-fame not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"[select] {exp_name}: loaded {len(df)} candidates from {path}")
    return df


def select_for_experiment(exp_name: str, complexity_budget: int) -> bool:
    df = _load_hof(exp_name)
    if df is None or len(df) == 0:
        print(f"[select] {exp_name}: no candidates — skipping.")
        return False

    layer  = config.get_layer(exp_name)
    opset  = config.get_opset(exp_name)
    in_dim = config.TRACE_INPUT_DIM[layer]

    # ── Identify key columns ──────────────────────────────────────────────────
    eq_col   = _find_col(df, _EQ_COLS)
    loss_col = _find_col(df, _LOSS_COLS)

    if eq_col is None:
        print(f"[select] {exp_name}: cannot find equation column in {list(df.columns)}")
        return False
    if loss_col is None:
        print(f"[select] {exp_name}: cannot find loss column; using first row.")

    # Ensure complexity column exists
    if "complexity" not in df.columns:
        df["complexity"] = range(1, len(df) + 1)

    # ── Filter by complexity budget ───────────────────────────────────────────
    feasible = df[df["complexity"] <= complexity_budget].copy()
    if feasible.empty:
        print(f"[select] {exp_name}: no equation within complexity {complexity_budget}; "
              f"relaxing to all candidates.")
        feasible = df.copy()

    # ── Compute operator cost for tie-breaking ────────────────────────────────
    def _cost(eq_str):
        try:
            info = count_ops(str(eq_str))
            return info["weighted_cost"]
        except Exception:
            return 9999

    feasible["_op_cost"] = feasible[eq_col].apply(_cost)

    # ── Sort: primary = fitness, secondary = _op_cost ────────────────────────
    # PySR may use "loss" (lower=better) or "score" ≈ -log(loss) (higher=better).
    if "loss" in feasible.columns:
        feasible = feasible.sort_values(["loss", "_op_cost"], ascending=True)
    elif "score" in feasible.columns:
        feasible = feasible.sort_values(["score", "_op_cost"], ascending=[False, True])
    else:
        print(f"[select] {exp_name}: no fitness column found; sorting by op cost only.")
        feasible = feasible.sort_values("_op_cost", ascending=True)

    best = feasible.iloc[0]
    eq_str     = str(best[eq_col])
    complexity = int(best.get("complexity", 0))
    loss_val   = float(best.get("loss", float("nan")))
    score_val  = float(best.get("score", float("nan")))

    op_info = {}
    try:
        op_info = count_ops(eq_str)
    except Exception as e:
        print(f"[select] {exp_name}: op_counter failed ({e})")

    result = {
        "experiment":     exp_name,
        "replacement":    layer,
        "operator_set":   opset,
        "equation":       eq_str,
        "complexity":     complexity,
        "loss":           loss_val,
        "score":          score_val,
        "input_dim":      in_dim,
        "selection_reason": (
            f"lowest loss ({loss_val:.6f}) within complexity budget {complexity_budget}"
        ),
        **op_info,
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    config.EQ_DIR.mkdir(parents=True, exist_ok=True)

    txt_path  = config.EQ_DIR / f"{exp_name}_best_equation.txt"
    json_path = config.EQ_DIR / f"{exp_name}_best_equation.json"

    with open(txt_path, "w") as f:
        f.write(eq_str + "\n")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[select] {exp_name}:")
    print(f"         equation   : {eq_str}")
    print(f"         complexity : {complexity}")
    print(f"         loss       : {loss_val:.6f}")
    print(f"         op cost    : {op_info.get('weighted_cost', '?')}")
    print(f"         saved to   : {json_path}")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Select best PySR equation per experiment")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--experiment",  type=str, choices=config.EXPERIMENTS)
    grp.add_argument("--all",         action="store_true")
    p.add_argument("--complexity-budget", type=int, default=config.PYSR_COMPLEXITY,
                   help="Maximum allowed equation complexity")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiments = config.EXPERIMENTS if args.all else [args.experiment]
    success = 0
    for exp in experiments:
        if select_for_experiment(exp, args.complexity_budget):
            success += 1
    print(f"\nSelected equations for {success}/{len(experiments)} experiments.")
