// baseline_fc3.cpp
// HLS C++ for baseline fc3 layer: dense 128 → 1
// This is the layer replaced by 1L symbolic experiments.
// Implements: logit = dot(x, w) + b

#include "baseline_fc3.h"

float baseline_fc3(float x[FC3_IN_DIM], float w[FC3_IN_DIM], float b) {
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_memory port=w
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=8

    float acc = b;
    for (int i = 0; i < FC3_IN_DIM; i++) {
#pragma HLS PIPELINE II=1
        acc += x[i] * w[i];
    }
    return acc;
}
