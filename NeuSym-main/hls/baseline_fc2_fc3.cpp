// baseline_fc2_fc3.cpp
// HLS C++ for baseline fc2 + fc3: dense 512 -> 128 (ReLU) -> 1
// Comparison target for 2L symbolic experiments.
//
// fc2: h2[i] = ReLU( B2[i] + sum_j( x[j] * W2[i][j] ) )
// fc3: logit  = B3 + sum_i( h2[i] * W3[i] )
//
// Weights are passed as arguments (not hardcoded) so the testbench can
// load trained values.  The previous version left W2/B2/W3 as uninitialized
// local arrays which produced garbage results.

#include "baseline_fc2_fc3.h"

float baseline_fc2_fc3(
    float x[FC2_IN_DIM],
    float W2[FC2_OUT_DIM][FC2_IN_DIM],
    float B2[FC2_OUT_DIM],
    float W3[FC2_OUT_DIM],
    float B3
) {
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_memory port=W2
#pragma HLS INTERFACE ap_memory port=B2
#pragma HLS INTERFACE ap_memory port=W3
// cyclic factor=16 on dim=2 of W2 and on x means 16 weight/input pairs are
// accessible per cycle, enabling 16-way parallel MACs in the inner loop.
#pragma HLS ARRAY_PARTITION variable=x  cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=W2 cyclic factor=16 dim=2

    float h2[FC2_OUT_DIM];
#pragma HLS ARRAY_PARTITION variable=h2 complete dim=1

    // fc2: 512 -> 128 with ReLU
    // PIPELINE on the inner loop; the cyclic partition of x and W2 lets HLS
    // schedule 16 multiplications per cycle automatically.
fc2_row:
    for (int i = 0; i < FC2_OUT_DIM; i++) {
        float acc = B2[i];
fc2_col:
        for (int j = 0; j < FC2_IN_DIM; j++) {
#pragma HLS PIPELINE II=1
            acc += x[j] * W2[i][j];
        }
        h2[i] = (acc > 0.0f) ? acc : 0.0f;
    }

    // fc3: 128 -> 1
    float logit = B3;
fc3:
    for (int i = 0; i < FC2_OUT_DIM; i++) {
#pragma HLS PIPELINE II=1
        logit += h2[i] * W3[i];
    }

    return logit;
}
