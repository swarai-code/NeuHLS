// SR-1L-SCE_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-1L-SCE
// Replacement: 1L   Operator set: SCE
// Equation: (((x121 + ((x24 - x0) + (x63 - x47))) + (-4.668614 - x66)) * 0.7971458) - (x38 * 2.8560774)

#include "SR-1L-SCE_symbolic_head.h"

float symbolic_head(float x[SYMBOLIC_DIM]) {
#pragma HLS INTERFACE ap_memory port=x

    const int idx[SYMBOLIC_NUM_TERMS] = {
        121, 24, 0, 63, 47, 66, 38
    };

    const float coef[SYMBOLIC_NUM_TERMS] = {
         0.7971458f,
         0.7971458f,
        -0.7971458f,
         0.7971458f,
        -0.7971458f,
        -0.7971458f,
        -2.8560774f
    };

#pragma HLS ARRAY_PARTITION variable=idx complete
#pragma HLS ARRAY_PARTITION variable=coef complete

    float acc = -4.668614f * 0.7971458f;

    for (int i = 0; i < SYMBOLIC_NUM_TERMS; i++) {
#pragma HLS PIPELINE II=1
        acc += coef[i] * x[idx[i]];
    }

    return acc;
}
