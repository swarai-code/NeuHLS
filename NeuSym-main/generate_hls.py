"""
generate_hls.py — Generate Vitis HLS C++ code for selected symbolic equations.

Outputs per experiment:
  outputs/hls/{exp}_symbolic_head.cpp
  outputs/hls/{exp}_symbolic_head.h

Also generates generic stubs in hls/ directory.
"""

from __future__ import annotations

import argparse
import ast as _ast
import json
import re
from pathlib import Path

import config

# ──────────────────────────────────────────────────────────────────────────────
# Equation → C++ translator
# ──────────────────────────────────────────────────────────────────────────────

def _get_used_indices(equation_str: str) -> list:
    """Return sorted list of variable indices that appear in the equation."""
    return sorted(set(int(m.group(1)) for m in re.finditer(r"x(\d+)", equation_str)))


def _sparse_params(indices: list) -> str:
    """Generate C function parameter list for only the used variables."""
    return ", ".join(f"float x_{i}" for i in indices)


def _to_cpp(equation_str: str, dim: int) -> str:
    """
    Convert a PySR equation string to a C++ expression.

    Transformations applied (in order):
      **             →  _square / powf  (Python AST handles arbitrary nesting)
      x0, x1, ...   →  x_0, x_1, ...  (named scalar params, not array indices)
      sin/cos/exp/sqrt/abs  →  sinf/cosf/expf/sqrtf/fabsf
      square/relu/max(x,0)  →  _square/_relu
      float literals        →  float32 (append f suffix)
    """
    expr = equation_str.strip()

    # ── Step 1: Replace ** with C++ equivalents via Python AST ────────────────
    # Using ast instead of regex correctly handles arbitrarily nested bases
    # such as ((x0 + x1) * x2)**2 where regex [^()]+ would fail.
    def _pow_to_cpp(base: str, exp_str: str) -> str:
        try:
            e = float(exp_str)
            if e == 2.0:   return f"_square({base})"
            if e == 3.0:   return f"(({base}) * ({base}) * ({base}))"
            if e == 0.5:   return f"sqrtf({base})"
            if e == -1.0:  return f"(1.0f / ({base}))"
            return f"powf({base}, {e}f)"
        except ValueError:
            return f"powf({base}, {exp_str})"

    while "**" in expr:
        try:
            tree = _ast.parse(expr, mode="eval")
        except SyntaxError:
            break
        pow_nodes = [
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.BinOp) and isinstance(n.op, _ast.Pow)
        ]
        if not pow_nodes:
            break
        # Process innermost ** first (no Pow in its direct children).
        leaf = next(
            (n for n in pow_nodes if not (
                (isinstance(n.left,  _ast.BinOp) and isinstance(n.left.op,  _ast.Pow)) or
                (isinstance(n.right, _ast.BinOp) and isinstance(n.right.op, _ast.Pow))
            )),
            pow_nodes[0],
        )
        base_str = expr[leaf.left.col_offset  : leaf.left.end_col_offset].strip()
        exp_str  = expr[leaf.right.col_offset : leaf.right.end_col_offset].strip()
        cpp      = _pow_to_cpp(base_str, exp_str)
        expr     = expr[:leaf.col_offset] + cpp + expr[leaf.end_col_offset:]

    # ── Step 2: Variable substitution: x123 → x_123 (named scalar param) ───────
    # Sort descending so x10 is replaced before x1 (word boundary \b also
    # protects against partial matches, but descending order is safer).
    indices = sorted(
        set(int(m.group(1)) for m in re.finditer(r"x(\d+)", expr)),
        reverse=True,
    )
    for i in indices:
        expr = re.sub(rf"\bx{i}\b", f"x_{i}", expr)

    # ── Step 3: Math function names ───────────────────────────────────────────
    expr = re.sub(r"\bsin\b",    "sinf",    expr)
    expr = re.sub(r"\bcos\b",    "cosf",    expr)
    expr = re.sub(r"\bexp\b",    "expf",    expr)
    expr = re.sub(r"\bsqrt\b",   "sqrtf",   expr)
    expr = re.sub(r"\babs\b",    "fabsf",   expr)
    expr = re.sub(r"\bsquare\b", "_square", expr)
    expr = re.sub(r"\brelu\b",   "_relu",   expr)
    expr = re.sub(r"\bmax\(([^,]+),\s*0(?:\.0)?\)", r"_relu(\1)", expr)

    # ── Step 4: Float literal promotion → float32 ────────────────────────────
    # Bare decimal literals like 2.1293101 are treated as double by the C
    # compiler, causing HLS to instantiate 64-bit FP units (expensive).
    # Appending 'f' forces float32 throughout the expression.
    expr = re.sub(r"(\d+\.\d+(?:[eE][+-]?\d+)?)(?!f)", r"\1f", expr)

    return expr


# ──────────────────────────────────────────────────────────────────────────────
# C++ / H template generators
# ──────────────────────────────────────────────────────────────────────────────

_CPP_TEMPLATE = """\
// {filename}
// Auto-generated HLS C++ for experiment: {exp_name}
// Replacement: {layer}   Operator set: {opset}
// Equation: {equation}
// Used inputs: {n_inputs} of {dim}

#include "{header}"
#include <math.h>

inline float _square(float v) {{ return v * v; }}
inline float _relu(float v)   {{ return (v > 0.0f) ? v : 0.0f; }}

// symbolic_head: sparse input interface — only the variables that appear in
// the equation are declared as parameters. This eliminates dangling ports,
// removes the input multiplexer, and reduces the FSM state count vs a full
// array interface. HLS maps each scalar float to an ap_none port automatically.
float symbolic_head({params}) {{
    return {cpp_expr};
}}
"""

_H_TEMPLATE = """\
// {filename}
// Auto-generated HLS header for experiment: {exp_name}
// Input dim: {dim}   Used inputs: {n_inputs}

#ifndef {guard}
#define {guard}

float symbolic_head({params});

#endif  // {guard}
"""

_TCL_TEMPLATE = """\
# run_hls.tcl — placeholder Vitis HLS synthesis script for {exp_name}
# Adjust paths and part number before running.

open_project hls_proj_{exp_name}
set_top symbolic_head
add_files {{{cpp_file}}}
open_solution "solution1"
set_part {{xc7z020clg484-1}}   ;# change to target device
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog
close_project
"""


def _generate_for_experiment(exp_name: str) -> bool:
    eq_path = config.EQ_DIR / f"{exp_name}_best_equation.json"
    if not eq_path.exists():
        print(f"[hls] {exp_name}: equation file not found ({eq_path}) — skipping.")
        return False

    with open(eq_path) as f:
        eq_data = json.load(f)

    equation_str = eq_data.get("equation", "x0")
    layer        = config.get_layer(exp_name)
    opset        = config.get_opset(exp_name)
    dim          = config.TRACE_INPUT_DIM[layer]

    used_indices = _get_used_indices(equation_str)
    params       = _sparse_params(used_indices)
    cpp_expr     = _to_cpp(equation_str, dim)

    stem    = f"{exp_name}_symbolic_head"
    cpp_fn  = f"{stem}.cpp"
    h_fn    = f"{stem}.h"
    tcl_fn  = f"{exp_name}_run_hls.tcl"
    guard   = f"{stem.upper().replace('-', '_')}_H"

    cpp_code = _CPP_TEMPLATE.format(
        filename  = cpp_fn,
        exp_name  = exp_name,
        layer     = layer,
        opset     = opset,
        equation  = equation_str,
        header    = h_fn,
        dim       = dim,
        n_inputs  = len(used_indices),
        params    = params,
        cpp_expr  = cpp_expr,
    )
    h_code = _H_TEMPLATE.format(
        filename  = h_fn,
        exp_name  = exp_name,
        guard     = guard,
        dim       = dim,
        n_inputs  = len(used_indices),
        params    = params,
    )
    tcl_code = _TCL_TEMPLATE.format(
        exp_name  = exp_name,
        cpp_file  = str(config.HLS_OUT_DIR / cpp_fn).replace("\\", "/"),
    )

    config.HLS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.HLS_OUT_DIR / cpp_fn).write_text(cpp_code)
    (config.HLS_OUT_DIR / h_fn).write_text(h_code)
    (config.HLS_OUT_DIR / tcl_fn).write_text(tcl_code)

    print(f"[hls] {exp_name}:  dim={dim}  eq={equation_str[:60]}...")
    print(f"       → {config.HLS_OUT_DIR / cpp_fn}")
    print(f"       → {config.HLS_OUT_DIR / h_fn}")
    return True


def _generate_generic_stubs():
    """Write generic stubs in hls/ for reference."""
    config.HLS_SRC_DIR.mkdir(parents=True, exist_ok=True)

    stub_cpp = """\
// hls/symbolic_head.cpp
// Generic stub — replace with a generated file from outputs/hls/
// after running: python generate_hls.py --all
//
// Generated files use a sparse scalar parameter interface:
//   float symbolic_head(float x_3, float x_17, float x_42) { return ...; }
// Only variables that appear in the equation are declared as parameters.
// No array interface and no ARRAY_PARTITION needed: each scalar float maps
// to its own ap_none port in HLS, eliminating the input mux entirely.
#include "symbolic_head.h"
#include <math.h>

inline float _square(float v) { return v * v; }
inline float _relu(float v)   { return (v > 0.0f) ? v : 0.0f; }

// Placeholder — copy the experiment-specific file from outputs/hls/ here
// and update the #include above to match its header.
float symbolic_head(float x_0) {
    return x_0;
}
"""
    stub_h = """\
// hls/symbolic_head.h
// Generic stub header — signature varies per experiment.
// Copy the matching header from outputs/hls/ for Vitis HLS synthesis.
//
// 1L experiments: h2 is 128-dim; equation uses a sparse subset of those inputs.
// 2L experiments: h1 is 512-dim; equation uses a sparse subset of those inputs.
// Both cases use scalar float parameters — no array, no ARRAY_PARTITION required.
#ifndef SYMBOLIC_HEAD_H
#define SYMBOLIC_HEAD_H

// Placeholder — replace with the exact signature from outputs/hls/*_symbolic_head.h
float symbolic_head(float x_0);

#endif  // SYMBOLIC_HEAD_H
"""
    stub_tcl = """\
# hls/run_hls.tcl
# Generic Vitis HLS synthesis script placeholder
# Copy a generated experiment CPP from outputs/hls/ and adjust below.

open_project symbolic_head_proj
set_top symbolic_head
add_files {symbolic_head.cpp}
open_solution "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog
close_project
"""
    (config.HLS_SRC_DIR / "symbolic_head.cpp").write_text(stub_cpp)
    (config.HLS_SRC_DIR / "symbolic_head.h").write_text(stub_h)
    (config.HLS_SRC_DIR / "run_hls.tcl").write_text(stub_tcl)
    print(f"[hls] Generic stubs written to {config.HLS_SRC_DIR}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate HLS C++ code for symbolic equations")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all",        action="store_true")
    grp.add_argument("--experiment", type=str, choices=config.EXPERIMENTS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _generate_generic_stubs()

    experiments = config.EXPERIMENTS if args.all else [args.experiment]
    ok = 0
    for exp in experiments:
        if _generate_for_experiment(exp):
            ok += 1
    print(f"\nGenerated HLS files for {ok}/{len(experiments)} experiments.")
