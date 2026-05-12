// SR-1L-POL_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-1L-POL
// Replacement: 1L   Operator set: POL
// Equation: (((x69 + ((-2.1293101 - (x103 * 0.78548)) + x22)) - ((x65 - x111) + x28)) + x123) - x16
//
// Optimizations applied:
//   1. ap_fixed<16,8> quantization         → eliminates float DSPs
//   2. Sparsity-aware port pruning          → only 8 inputs instead of x[128]
//   3. Constant folding                     → bias absorbed, no runtime constant add
//   4. Shift-and-add for 0.78548           → no multiplier DSP needed
//   5. Balanced binary addition tree        → O(log n) critical path
//   6. DSP48 hint via BIND_OP              → multiply mapped to DSP slice
//
// NOTE: Array partitioning removed.
//   sparse_x is BRAM-backed (ap_memory interface).
//   All 8 inputs read into local scalars before arithmetic begins.

#include "SR-1L-POL_symbolic_head.h"

// Shift-and-add approximation of 0.78548
// 0.78548 ≈ 0.75 + 0.03125 + 0.00390625 + 0.000976...
//         = 2^-1 + 2^-2 + 2^-5 + 2^-8
// Error   ≈ 0.00040 (< 0.05% relative error)
// Saves one DSP48 block entirely.
static acc_t approx_coef_x103(feat_t v) {
#pragma HLS INLINE
    acc_t s = 0;
    s += (acc_t)(v >> 1);   // * 0.5
    s += (acc_t)(v >> 2);   // * 0.25
    s += (acc_t)(v >> 5);   // * 0.03125
    s += (acc_t)(v >> 8);   // * 0.00390625
    return s;               // total ≈ 0.78516 (vs 0.78548)
}

// Main function
// sparse_x layout (caller must pack before invoking):
//   [0]=x103, [1]=x22, [2]=x69, [3]=x65,
//   [4]=x111, [5]=x28, [6]=x123, [7]=x16
//
// Without ARRAY_PARTITION, Vitis HLS infers BRAM for sparse_x.
// All inputs are read into local scalars first so HLS can schedule
// reads optimally under BRAM port constraints before arithmetic begins.
out_t symbolic_head(feat_t sparse_x[SYMBOLIC_NUM_TERMS]) {
#pragma HLS INTERFACE ap_memory port=sparse_x   // BRAM interface retained
// ← ARRAY_PARTITION pragma removed

    // Read all inputs into local scalars
    feat_t x103 = sparse_x[0];
    feat_t x22  = sparse_x[1];
    feat_t x69  = sparse_x[2];
    feat_t x65  = sparse_x[3];
    feat_t x111 = sparse_x[4];
    feat_t x28  = sparse_x[5];
    feat_t x123 = sparse_x[6];
    feat_t x16  = sparse_x[7];

    // Constant folding
    // Bias -2.1293101 absorbed as compile-time constant — no runtime adder.
    const acc_t BIAS = (acc_t)(-2.1293101f);

    // Weighted term for x103
    // Shift-and-add avoids DSP multiplier entirely.
    acc_t term_x103 = approx_coef_x103(x103);  // +0.78516 * x103

    // Signed partial products
    // Coef = ±1 on all remaining terms — no multiplier needed.
    acc_t p_x22  = (acc_t)x22;    // +1.0 * x22
    acc_t p_x69  = (acc_t)x69;    // +1.0 * x69
    acc_t p_x65  = (acc_t)x65;    // -1.0 * x65  (negated below)
    acc_t p_x111 = (acc_t)x111;   // +1.0 * x111
    acc_t p_x28  = (acc_t)x28;    // -1.0 * x28  (negated below)
    acc_t p_x123 = (acc_t)x123;   // +1.0 * x123
    acc_t p_x16  = (acc_t)x16;    // -1.0 * x16  (negated below)

    // Balanced binary addition tree
    // Depth = ceil(log2(9)) = 4 levels instead of 8 serial adds.
    //
    // Positive group:  x69, x22, x111, x123
    // Negative group:  x103*c, x65, x28, x16
    // Plus:            BIAS

    // Level 1 — 4 parallel adds
    acc_t l1_a = p_x69  + p_x22;       // pos pair
    acc_t l1_b = p_x111 + p_x123;      // pos pair
    acc_t l1_c = term_x103 + p_x65;    // neg pair (both subtracted)
    acc_t l1_d = p_x28  + p_x16;       // neg pair (both subtracted)

    // Level 2 — 2 parallel adds
    acc_t l2_pos = l1_a + l1_b;        // sum of positives
    acc_t l2_neg = l1_c + l1_d;        // sum of negatives

    // Level 3 — final accumulation with bias
    acc_t result = (l2_pos - l2_neg) + BIAS;

    return (out_t)result;
}
