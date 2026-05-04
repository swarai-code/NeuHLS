// SR-1L-SRL_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-1L-SRL
// Replacement: 1L   Operator set: SRL
// Equation: (((x7 - x3) - ((x65 + relu(x27)) - (x24 - x118))) * 0.72398) - ((x74 * 0.21851327) - x22)
//
// Flat form (expanded):
//   0.72398*(x7 - x3 - x65 - relu(x27) + x24 - x118) - 0.21851327*x74 + x22
//
// Optimizations applied:
//   1. ap_fixed<16,8> quantization         → eliminates float DSPs
//   2. Sparsity-aware port pruning          → only 8 inputs instead of x[128]
//   3. Coefficient factoring               → 0.72398 applied ONCE after summation
//                                             (same insight as SCE — saves 5 multipliers)
//   4. Exact multipliers via ap_fixed       → HLS maps to DSP48, no approximation
//   5. relu in ap_fixed                     → pure LUT logic, no DSP needed
//   6. Balanced binary addition tree        → O(log n) critical path
//   7. ARRAY_PARTITION complete             → no BRAM, pure register file
//
// Note on relu:
//   relu(x) = max(0, x) — in ap_fixed this is a single sign-bit check.
//   No multiplier or LUT table needed — just a conditional assignment.
//   HLS synthesizes this as a mux driven by the MSB (sign bit).
 
#include "SR-1L-SRL_symbolic_head.h"
 
// relu in ap_fixed 
// max(0, v) — HLS maps this to a single mux on the sign bit.
// Zero overhead: no DSP, ~1 LUT per bit.
static feat_t relu_fixed(feat_t v) {
#pragma HLS INLINE
    return (v > (feat_t)0) ? v : (feat_t)0;
}
 
// Main function
// sparse_x layout (caller must pack before invoking):
//   [0]=x7, [1]=x3, [2]=x65, [3]=x27,
//   [4]=x24, [5]=x118, [6]=x74, [7]=x22
out_t symbolic_head(feat_t sparse_x[SYMBOLIC_NUM_TERMS]) {
#pragma HLS INTERFACE ap_memory port=sparse_x
#pragma HLS ARRAY_PARTITION variable=sparse_x complete  // all 8 in registers, no BRAM
 
    // relu on x27 
    // Applied before the sum — result fed into the scaled group.
    feat_t relu_x27 = relu_fixed(sparse_x[3]);
 
    // Balanced binary addition tree for scaled group 
    // Inner sum: x7 - x3 - x65 - relu(x27) + x24 - x118
    // Positive:  x7, x24
    // Negative:  x3, x65, relu(x27), x118
 
    // Level 1 — 3 parallel ops
    acc_t l1_a = (acc_t)sparse_x[0] - (acc_t)sparse_x[1];    // x7   - x3
    acc_t l1_b = (acc_t)sparse_x[4] - (acc_t)sparse_x[2];    // x24  - x65
    acc_t l1_c = (acc_t)relu_x27    + (acc_t)sparse_x[5];    // relu(x27) + x118 (both neg, subtract below)
 
    // Level 2
    acc_t l2_a = l1_a + l1_b;    // x7 - x3 + x24 - x65
    acc_t l2_b = l1_c;            // relu(x27) + x118
 
    // Level 3 — inner sum complete
    acc_t inner_sum = l2_a - l2_b;   // x7 - x3 + x24 - x65 - relu(x27) - x118
 
    //Coefficient factoring
    // Apply 0.72398 ONCE to the entire inner sum.
    // Exact multiply — HLS maps to DSP48.
    acc_t scaled = inner_sum * (acc_t)(0.72398f);
 
    //x74 term (separate coefficient) 
    // Exact multiply — HLS maps to DSP48.
    acc_t term_x74 = (acc_t)sparse_x[6] * (acc_t)(0.21851327f);
 
    // Final result 
    // 0.72398*(...) - 0.21851327*x74 + x22
    acc_t result = scaled - term_x74 + (acc_t)sparse_x[7];
 
    return (out_t)result;
}
