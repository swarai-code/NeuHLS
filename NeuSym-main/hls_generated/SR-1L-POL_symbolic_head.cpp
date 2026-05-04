// SR-1L-POL_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-1L-POL
// Replacement: 1L   Operator set: POL
// Equation: (((x69 + ((-2.1293101 - (x103 * 0.78548)) + x22)) - ((x65 - x111) + x28)) + x123) - x16

#include "SR-1L-POL_symbolic_head.h"

float symbolic_head(float x[SYMBOLIC_DIM]) {
#pragma HLS INTERFACE ap_memory port=x

    const int idx[SYMBOLIC_NUM_TERMS] = {
        103, 22, 69, 65, 111, 28, 123, 16
    };

    const float coef[SYMBOLIC_NUM_TERMS] = {
        -0.78548f, 1.0f, 1.0f, -1.0f,
         1.0f,   -1.0f, 1.0f, -1.0f
    };

#pragma HLS ARRAY_PARTITION variable=idx complete
#pragma HLS ARRAY_PARTITION variable=coef complete

    float acc = -2.1293101f;

    for (int i = 0; i < SYMBOLIC_NUM_TERMS; i++) {
#pragma HLS PIPELINE II=1
        acc += coef[i] * x[idx[i]];
    }

    return acc;
}
