// baseline_fc3.cpp
// HLS C++ for baseline fc3 layer: dense 128 → 1
// This is the layer replaced by 1L symbolic experiments.
// Implements: logit = dot(x, w) + b
//
// Changes for fair comparison with SR-1L-* symbolic experiments:
//   1. float → ap_fixed<16,8>       same precision as all symbolic heads
//   2. acc_t wider (24,10)          same accumulator width to prevent overflow
//   3. cyclic factor=8 kept         baseline is genuinely dense — cannot prune
//   4. PIPELINE II=1 kept           fair: baseline needs the loop, symbolic don't
//   5. No structural changes        baseline dot-product cannot be factored
//
// What is intentionally NOT changed:
//   - The loop stays — baseline is dense (128 mults), symbolic heads are sparse (7-8)
//   - No port pruning — baseline genuinely uses all 128 inputs
//   - No coefficient factoring — weights are all different values
//
// This means the baseline is fairly penalized for being dense,
// which is exactly the point of the symbolic replacement comparison.

#include "baseline_fc3.h"

out_t baseline_fc3(feat_t x[FC3_IN_DIM], feat_t w[FC3_IN_DIM], acc_t b) {
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_memory port=w
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=8

    acc_t acc = b;
    for (int i = 0; i < FC3_IN_DIM; i++) {
#pragma HLS PIPELINE II=1
        acc += (acc_t)x[i] * (acc_t)w[i];
    }
    return (out_t)acc;
}
