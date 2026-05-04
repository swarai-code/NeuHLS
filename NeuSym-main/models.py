"""
models.py — BaselineMLP, Hybrid1L, Hybrid2L, and symbolic equation evaluator.
"""

from __future__ import annotations

import ast as _ast
import math
import re
import warnings
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import config


# ──────────────────────────────────────────────────────────────────────────────
# Baseline MLP
# ──────────────────────────────────────────────────────────────────────────────

class BaselineMLP(nn.Module):
    """
    3072 → FC(512) → ReLU → FC(128) → ReLU → FC(1)
    Optionally returns intermediate activations h1, h2.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(config.INPUT_DIM, config.FC1_DIM)
        self.fc2 = nn.Linear(config.FC1_DIM,   config.FC2_DIM)
        self.fc3 = nn.Linear(config.FC2_DIM,   config.OUTPUT_DIM)

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        h1    = torch.relu(self.fc1(x))    # [B, 512]
        h2    = torch.relu(self.fc2(h1))   # [B, 128]
        logit = self.fc3(h2)               # [B, 1]

        if return_activations:
            return logit, h1, h2
        return logit


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic equation evaluator
# ──────────────────────────────────────────────────────────────────────────────

def _safe_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


def _safe_square(x: torch.Tensor) -> torch.Tensor:
    return x * x


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Avoid near-zero division on both sides of zero.
    # sign(0) is mapped to +1 so the denominator stays positive when b==0.
    sign = torch.sign(b)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return a / (b.abs().clamp(min=1e-7) * sign)


class _SafeDivTransformer(_ast.NodeTransformer):
    """Replace every infix / with a _safe_div() call so near-zero
    denominators are handled smoothly instead of producing NaN/Inf."""
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, _ast.Div):
            return _ast.Call(
                func=_ast.Name(id="_safe_div", ctx=_ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        return node


def _preprocess_expr(eq_str: str) -> str:
    """Rewrite infix / as _safe_div() so the eval namespace can intercept it.
    Falls back to the original string on Python < 3.9 or a SyntaxError."""
    try:
        tree = _ast.parse(eq_str, mode="eval")
        tree = _SafeDivTransformer().visit(tree)
        _ast.fix_missing_locations(tree)
        return _ast.unparse(tree)          # ast.unparse requires Python ≥ 3.9
    except (SyntaxError, AttributeError):
        return eq_str


def _make_torch_evaluator(equation_str: str, n_inputs: int) -> Callable:
    """
    Build a batched PyTorch callable from a PySR equation string.

    Variables must be named x0, x1, ..., x{n_inputs-1}.
    Supported ops: +, -, *, /, square, relu, sin, cos, exp.

    Returns a function f(h: Tensor[B, n_inputs]) -> Tensor[B, 1].
    """
    expr = _preprocess_expr(equation_str.strip())

    def evaluator(h: torch.Tensor) -> torch.Tensor:
        """h: [B, n_inputs] → [B, 1]"""
        # Build per-variable slices so the expression can reference them
        local_vars = {f"x{i}": h[:, i] for i in range(n_inputs)}
        local_ops = {
            "square":    _safe_square,
            "relu":      _safe_relu,
            "_safe_div": _safe_div,   # inserted by _preprocess_expr for infix /
            "sin":       torch.sin,
            "cos":       torch.cos,
            "exp":       torch.exp,
            "sqrt":      torch.sqrt,
            "abs":       torch.abs,
            "log":       torch.log,
            "__builtins__": {},
        }
        local_vars.update(local_ops)
        try:
            result = eval(expr, {"__builtins__": {}}, local_vars)  # noqa: S307
        except Exception as exc:
            warnings.warn(f"Symbolic evaluator error: {exc}. Falling back to zero.")
            result = torch.zeros(h.shape[0], device=h.device)

        if isinstance(result, torch.Tensor):
            # Clamp NaN/Inf that can arise from division-by-zero in PySR equations.
            result = torch.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
            return result.unsqueeze(-1).float()
        # Scalar constant equation
        return torch.full((h.shape[0], 1), float(result),
                          device=h.device, dtype=torch.float32)

    return evaluator


def build_symbolic_evaluator(equation_str: str, n_inputs: int) -> Callable:
    """
    Public factory.  Tries PyTorch evaluator; falls back gracefully.
    """
    evaluator = _make_torch_evaluator(equation_str, n_inputs)
    return evaluator


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid models
# ──────────────────────────────────────────────────────────────────────────────

class Hybrid1L(nn.Module):
    """
    Replaces fc3 with a symbolic equation.

    Forward: x → fc1 → ReLU → fc2 → ReLU → symbolic_fc3(h2) → logit
    """

    def __init__(self, baseline: BaselineMLP, equation_str: str):
        super().__init__()
        # Intentional reference share — fc1/fc2 weights are frozen (eval mode only).
        self.fc1 = baseline.fc1
        self.fc2 = baseline.fc2
        self._equation_str = equation_str
        self._evaluator = build_symbolic_evaluator(equation_str, config.FC2_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1    = torch.relu(self.fc1(x))
        h2    = torch.relu(self.fc2(h1))
        logit = self._evaluator(h2)    # [B, 1]
        return logit

    @property
    def equation(self) -> str:
        return self._equation_str


class Hybrid2L(nn.Module):
    """
    Replaces fc2 + fc3 with one symbolic equation.

    Forward: x → fc1 → ReLU → symbolic_2L_head(h1) → logit
    """

    def __init__(self, baseline: BaselineMLP, equation_str: str):
        super().__init__()
        # Intentional reference share — fc1 weight is frozen (eval mode only).
        self.fc1 = baseline.fc1
        self._equation_str = equation_str
        self._evaluator = build_symbolic_evaluator(equation_str, config.FC1_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1    = torch.relu(self.fc1(x))
        logit = self._evaluator(h1)    # [B, 1]
        return logit

    @property
    def equation(self) -> str:
        return self._equation_str


# ──────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ──────────────────────────────────────────────────────────────────────────────

def load_baseline(checkpoint_path, device="cpu") -> BaselineMLP:
    model = BaselineMLP()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Support both raw state-dict and wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_hybrid(
    baseline: BaselineMLP,
    equation_str: str,
    layer: str,           # "1L" or "2L"
) -> nn.Module:
    if layer == "1L":
        return Hybrid1L(baseline, equation_str)
    if layer == "2L":
        return Hybrid2L(baseline, equation_str)
    raise ValueError(f"Unknown layer: {layer!r}")
