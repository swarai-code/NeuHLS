"""
config.py — Central hyperparameter and path configuration for test_svhn pipeline.
"""

from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 20
try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    del _torch
except ImportError:
    DEVICE = "cpu"

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent
DATA_DIR     = ROOT_DIR / "data"
OUTPUT_DIR   = ROOT_DIR / "outputs"
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
TRACE_DIR    = OUTPUT_DIR / "traces"
PYSR_DIR     = OUTPUT_DIR / "pysr"
EQ_DIR       = OUTPUT_DIR / "equations"
HLS_OUT_DIR  = OUTPUT_DIR / "hls"
RESULTS_DIR  = OUTPUT_DIR / "results"
HLS_SRC_DIR  = ROOT_DIR / "hls"

# ── Model architecture ─────────────────────────────────────────────────────────
INPUT_DIM  = 3072   # 32×32×3 flattened
FC1_DIM    = 512
FC2_DIM    = 128
OUTPUT_DIM = 1

# ── SVHN normalisation (per-channel mean / std on the full training set) ───────
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD  = (0.1980, 0.2010, 0.1970)

# ── Binary-classification label mapping ───────────────────────────────────────
KEEP_DIGITS   = [1, 7]          # only these SVHN labels are used
LABEL_MAP     = {1: 0, 7: 1}   # digit-1 → 0,  digit-7 → 1

# ── PySR ──────────────────────────────────────────────────────────────────────
PYSR_NITERATIONS  = 100
PYSR_POPULATIONS  = 20
PYSR_MAXSIZE      = 20          # PySR tree-size limit (controls search space during fit)
PYSR_COMPLEXITY   = 20          # complexity budget used in select_equation.py to filter the hall-of-fame
PYSR_TIMEOUT_MINS = None        # set to e.g. 10 to cap runtime (minutes)
PYSR_SEED         = SEED

# ── Operator-set definitions (PySR format) ─────────────────────────────────────
OPERATOR_SETS = {
    "POL": {
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators":  [],
    },
    "SRL": {
        "binary_operators": ["+", "-", "*"],
        "unary_operators":  [
            "square(x::T) where {T} = x*x",
            "relu(x::T) where {T} = x > zero(T) ? x : zero(T)",
        ],
    },
    "SCE": {
        "binary_operators": ["+", "-", "*"],
        "unary_operators":  ["sin", "cos", "exp"],
    },
}

# ── Experiment registry ────────────────────────────────────────────────────────
EXPERIMENTS = [
    "SR-1L-POL",
    "SR-1L-SRL",
    "SR-1L-SCE",
    "SR-2L-POL",
    "SR-2L-SRL",
    "SR-2L-SCE",
]

def get_opset(experiment: str) -> str:
    """Return 'POL', 'SRL', or 'SCE' for the given experiment name."""
    for tag in ("POL", "SRL", "SCE"):
        if tag in experiment:
            return tag
    raise ValueError(f"Cannot determine operator set from experiment: {experiment}")

def get_layer(experiment: str) -> str:
    """Return '1L' or '2L' for the given experiment name."""
    if "1L" in experiment:
        return "1L"
    if "2L" in experiment:
        return "2L"
    raise ValueError(f"Cannot determine layer from experiment: {experiment}")

# ── Symbolic operator cost model ───────────────────────────────────────────────
OP_COSTS = {
    "add":      1,
    "sub":      1,
    "mul":      2,
    "div":      8,
    "square":   2,
    "relu":     1,
    "sin":      8,
    "cos":      8,
    "exp":     12,
}

# ── Baseline MAC counts ────────────────────────────────────────────────────────
FC1_MACS = INPUT_DIM * FC1_DIM        # 3072 × 512 = 1,572,864
FC2_MACS = FC1_DIM   * FC2_DIM        # 512  × 128 =    65,536
FC3_MACS = FC2_DIM   * OUTPUT_DIM     # 128  × 1   =       128
BASELINE_MACS  = FC1_MACS + FC2_MACS + FC3_MACS
BASELINE_FLOPS = 2 * BASELINE_MACS    # multiply-accumulate counts as 2 FLOPs

# Replaced MACs per replacement type
REPLACED_MACS = {
    "1L": FC3_MACS,               # only fc3 replaced
    "2L": FC2_MACS + FC3_MACS,    # fc2 + fc3 replaced
}

KEPT_MACS = {
    "1L": FC1_MACS + FC2_MACS,
    "2L": FC1_MACS,
}

# ── Input dimension per experiment type ───────────────────────────────────────
TRACE_INPUT_DIM = {
    "1L": FC2_DIM,   # h2 shape [N, 128]
    "2L": FC1_DIM,   # h1 shape [N, 512]
}
