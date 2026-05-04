"""
op_counter.py — Count symbolic operators in a PySR equation string.

Uses SymPy for parsing. Variable names are x0, x1, ..., x{N-1}.
Also handles custom ops: square, relu.

Cost model (from config.OP_COSTS):
  add/sub: 1,  mul: 2,  div: 8,  square: 2,  relu: 1,
  sin: 8,  cos: 8,  exp: 12
"""

from __future__ import annotations

import re
from typing import Any

import sympy as sp
from sympy import (Add, Mul, Pow, sin, cos, exp, Abs, sqrt,
                   Symbol, Number, Integer, Float, Rational)

import config


# ── Custom SymPy function stubs (so lambdify / parsing works) ──────────────────

class square(sp.Function):  # noqa: N801
    """square(x) = x**2 — kept unevaluated so _count_nodes can detect it.
    No eval() override: if SymPy auto-simplifies, the square branch becomes dead code."""
    is_real = True

class relu(sp.Function):    # noqa: N801
    """relu(x) = max(x, 0) — not simplifiable symbolically."""
    pass


# ── Parser ─────────────────────────────────────────────────────────────────────

_SYMPY_LOCALS = {
    "square": square,
    "relu":   relu,
    "sin":    sp.sin,
    "cos":    sp.cos,
    "exp":    sp.exp,
    "sqrt":   sp.sqrt,
    "abs":    sp.Abs,
    "Abs":    sp.Abs,
}


def _parse(equation_str: str) -> sp.Expr:
    """Parse a PySR equation string into a SymPy expression."""
    # Replace max(x, 0) → relu(x)
    eq = re.sub(r"max\(([^,]+),\s*0(?:\.0)?\)", r"relu(\1)", equation_str)

    # Build local namespace with auto-generated variable symbols
    var_re = re.compile(r"x(\d+)")
    indices = set(int(m.group(1)) for m in var_re.finditer(eq))
    sym_locals = {f"x{i}": sp.Symbol(f"x{i}") for i in indices}
    sym_locals.update(_SYMPY_LOCALS)

    try:
        expr = sp.sympify(eq, locals=sym_locals)
    except Exception as e:
        raise ValueError(f"Cannot parse equation: {equation_str!r}  ({e})")
    return expr


def _is_neg(node) -> bool:
    """True for Mul(-1, x) — SymPy's canonical representation of unary negation."""
    return (
        isinstance(node, Mul)
        and len(node.args) == 2
        and isinstance(node.args[0], (Integer, Float, Rational))
        and float(node.args[0]) == -1.0
    )


def _count_nodes(expr: sp.Expr) -> dict[str, int]:
    """Walk the SymPy expression tree and count operator types."""
    counts = {k: 0 for k in
              ("add", "sub", "mul", "div", "square", "relu",
               "sin", "cos", "exp", "pow", "other")}

    def _walk(node):
        if isinstance(node, Add):
            args = node.args
            # SymPy represents a-b as Add(a, Mul(-1,b)).
            # Count Mul(-1,x) terms as subtractions, not as add+multiply.
            n_neg = sum(1 for a in args if _is_neg(a))
            n_pos = len(args) - n_neg
            counts["add"] += max(n_pos - 1, 0)
            counts["sub"] += n_neg
            for a in args:
                if _is_neg(a):
                    _walk(a.args[1])  # skip the -1 factor, walk the inner expr
                else:
                    _walk(a)

        elif isinstance(node, Mul):
            # Pure negation Mul(-1, x): cost is 0; already counted as a
            # subtraction by the parent Add handler (or free if standalone).
            if _is_neg(node):
                _walk(node.args[1])
                return

            args = node.args
            # Check for division: presence of Pow(x, -1) or Pow(x, -n)
            n_div, n_mul = 0, 0
            for a in args:
                if isinstance(a, Pow) and isinstance(a.exp, (Integer, Float, Rational)):
                    if float(a.exp) < 0:
                        n_div += 1
                        _walk(a.args[0])   # walk the denominator base so its ops are counted
                    else:
                        n_mul += 1
                        _walk(a)       # handle things like x**2 inside Mul
                else:
                    n_mul += 1
                    _walk(a)
            # Each Mul between two things is one operation
            total_terms = len(args)
            counts["mul"] += max(total_terms - 1 - n_div, 0)
            counts["div"] += n_div

        elif isinstance(node, Pow):
            base, exp_val = node.args
            if isinstance(exp_val, (Integer, Float, Rational)):
                fexp = float(exp_val)
                if fexp == 2.0:
                    counts["square"] += 1
                elif fexp == -1.0:
                    counts["div"] += 1
                elif fexp > 0:
                    counts["mul"] += max(int(fexp) - 1, 0)
                else:
                    counts["div"] += 1
            else:
                counts["pow"] += 1
            _walk(base)

        elif isinstance(node, square):
            counts["square"] += 1
            for a in node.args:
                _walk(a)

        elif isinstance(node, relu):
            counts["relu"] += 1
            for a in node.args:
                _walk(a)

        elif isinstance(node, sp.sin):
            counts["sin"] += 1
            for a in node.args:
                _walk(a)

        elif isinstance(node, sp.cos):
            counts["cos"] += 1
            for a in node.args:
                _walk(a)

        elif isinstance(node, sp.exp):
            counts["exp"] += 1
            for a in node.args:
                _walk(a)

        elif isinstance(node, (Symbol, Number, Integer, Float, Rational)):
            pass   # leaf — no cost

        else:
            # Generic function application
            counts["other"] += 1
            for a in node.args:
                _walk(a)

    _walk(expr)
    return counts


def count_ops(equation_str: str) -> dict[str, Any]:
    """
    Parse equation_str and return an op-count/cost summary dict.

    Keys:
      counts          – {op_name: count}
      weighted_cost   – scalar cost using config.OP_COSTS
      total_ops       – sum of all operator counts
      has_expensive   – bool (any op with cost >= 8 present)
      equation_str    – original string
    """
    expr   = _parse(equation_str)
    counts = _count_nodes(expr)

    cost_map = {
        "add":    config.OP_COSTS["add"],
        "sub":    config.OP_COSTS["sub"],
        "mul":    config.OP_COSTS["mul"],
        "div":    config.OP_COSTS["div"],
        "square": config.OP_COSTS["square"],
        "relu":   config.OP_COSTS["relu"],
        "sin":    config.OP_COSTS["sin"],
        "cos":    config.OP_COSTS["cos"],
        "exp":    config.OP_COSTS["exp"],
    }

    weighted = sum(counts.get(op, 0) * cost for op, cost in cost_map.items())
    total    = sum(counts.values())
    expensive_ops = {"div", "sin", "cos", "exp"}
    has_exp  = any(counts.get(op, 0) > 0 for op in expensive_ops)

    return {
        "counts":        counts,
        "weighted_cost": weighted,
        "total_ops":     total,
        "has_expensive": has_exp,
        "equation_str":  equation_str,
    }


if __name__ == "__main__":
    # Quick self-test
    examples = [
        "x0 + x1 * x2",
        "sin(x0) + cos(x1)",
        "relu(x0 - x1) * x2",
        "square(x0) + x1 / x2",
        "exp(x0) - 3.5",
        "x0",
    ]
    for eq in examples:
        info = count_ops(eq)
        print(f"{eq!r:45s}  cost={info['weighted_cost']:4d}  "
              f"ops={info['total_ops']}  expensive={info['has_expensive']}")
