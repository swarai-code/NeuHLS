// SR-2L-SRL_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-2L-SRL
// Replacement: 2L   Operator set: SRL
// Equation: (x63 + ((x233 - x222) + ((((x135 - relu(x399 - x44)) - x300) - x181) - x301))) * 0.95504266

#include "SR-2L-SRL_symbolic_head.h"
#include <math.h>

inline float _square(float v) { return v * v; }
inline float _relu(float v)   { return (v > 0.0f) ? v : 0.0f; }

// symbolic_head: pure combinational function (no loops).
// ARRAY_PARTITION complete makes every element available in the same clock
// cycle, which is required when the expression references many inputs at once.
// For dim > 128 (2L experiments) the tool may warn about register pressure;
// lower the partition factor or switch to ap_fifo streaming if needed.
float symbolic_head(float x[512]) {
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=16 dim=1
    return (x[63] + ((x[233] - x[222]) + ((((x[135] - _relu(x[399] - x[44])) - x[300]) - x[181]) - x[301]))) * 0.95504266;
}
