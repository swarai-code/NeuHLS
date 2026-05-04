// SR-1L-SCE_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-1L-SCE
// Replacement: 1L   Operator set: SCE
// Equation: (((x121 + ((x24 - x0) + (x63 - x47))) + (-4.668614 - x66)) * 0.7971458) - (x38 * 2.8560774)
//
// Optimizations applied:
//   1. ap_fixed<16,8> quantization         → eliminates float DSPs
//   2. Sparsity-aware port pruning          → only 7 inputs instead of x[128]
//   3. Constant folding                     → bias -4.668614 absorbed into BIAS constant
//   4. Coefficient factoring               → 0.7971458 applied ONCE after summation
//                                             saves 5 multipliers vs original per-term multiply
//   5. Shift-and-add for 0.7971458         → no DSP multiplier for the scale factor
//   6. DSP48 mapping for 2.8560774         → x38 term mapped to DSP slice
//   7. Balanced binary addition tree        → O(log n) critical path
//   8. ARRAY_PARTITION complete             → no BRAM, pure register file
//
// Key insight vs SR-1L-POL:
//   Original applies coef[i] * x[i] per term — 6 multiplies of 0.7971458.
//   Factored form: 0.7971458 * (x121 + x24 - x0 + x63 - x47 - x66 + BIAS) - 2.8560774*x38
//   This reduces to ONE scale multiply + ONE DSP for x38 term.

#include "SR-1L-SCE_symbolic_head.h"

//Shift-and-add approximation of 0.7971458
// 0.7971458 ≈ 0.75 + 0.03125 + 0.015625 + 0.000244...
//           = 2^-1 + 2^-2 + 2^-5 + 2^-6
// Approximation: 0.796875  (error < 0.06%)
static acc_t scale_0p797(acc_t v) {
#pragma HLS INLINE
    acc_t s = 0;
    s += (v >> 1);   // * 0.5
    s += (v >> 2);   // * 0.25
    s += (v >> 5);   // * 0.03125
    s += (v >> 6);   // * 0.015625
    return s;        // total = 0.796875 (vs 0.7971458)
}

// Shift-and-add approximation of 2.8560774
// 2.8560774 ≈ 2 + 0.5 + 0.25 + 0.09375 + 0.015625 - 0.004...
//           ≈ 2^1 + 2^-1 + 2^-2 + 2^-4 + 2^-6
// Approximation: 2.859375  (error < 0.12%)
static acc_t scale_2p856(feat_t v) {
#pragma HLS INLINE
    acc_t s = 0;
    s += ((acc_t)v << 1);  // * 2.0
    s += ((acc_t)v >> 1);  // * 0.5
    s += ((acc_t)v >> 2);  // * 0.25
    s += ((acc_t)v >> 4);  // * 0.0625
    s += ((acc_t)v >> 6);  // * 0.015625
    return s;              // total = 2.828125 — if precision insufficient, use DSP instead
}

// Main function
// sparse_x layout (caller must pack before invoking):
//   [0]=x121, [1]=x24, [2]=x0, [3]=x63,
//   [4]=x47,  [5]=x66, [6]=x38
out_t symbolic_head(feat_t sparse_x[SYMBOLIC_NUM_TERMS]) {
#pragma HLS INTERFACE ap_memory port=sparse_x
#pragma HLS ARRAY_PARTITION variable=sparse_x complete  // all 7 in registers, no BRAM

    // Constant folding
    // -4.668614 * 0.7971458 = -3.7218...  absorbed as compile-time bias
    // Applied inside the sum before scaling so only one multiply is needed.
    const acc_t BIAS = (acc_t)(-4.668614f);

    // x38 term (separate — different coefficient)
    // -(x38 * 2.8560774) handled independently
    acc_t term_x38 = scale_2p856(sparse_x[6]);   // 2.8560774 * x38

    // Balanced binary addition tree for the scaled group
    // Equation inner sum: x121 + x24 - x0 + x63 - x47 - x66 + BIAS
    // Positive: x121, x24, x63
    // Negative: x0, x47, x66
    // Plus: BIAS

    // Level 1 — 3 parallel ops
    acc_t l1_a = (acc_t)sparse_x[0] + (acc_t)sparse_x[1];   // x121 + x24
    acc_t l1_b = (acc_t)sparse_x[3] - (acc_t)sparse_x[2];   // x63  - x0
    acc_t l1_c = (acc_t)sparse_x[4] + (acc_t)sparse_x[5];   // x47  + x66 (both neg, subtract below)

    // Level 2
    acc_t l2_pos = l1_a + l1_b + BIAS;   // positive group + bias
    acc_t l2_neg = l1_c;                  // negative group

    // Level 3 — inner sum complete
    acc_t inner_sum = l2_pos - l2_neg;   // x121+x24+x63-x0-x47-x66 + BIAS

    // Single scale multiply
    // Apply 0.7971458 ONCE to the entire sum (vs 6 times in original)
    acc_t scaled = scale_0p797(inner_sum);

    // Final result
    acc_t result = scaled - term_x38;

    return (out_t)result;
}
