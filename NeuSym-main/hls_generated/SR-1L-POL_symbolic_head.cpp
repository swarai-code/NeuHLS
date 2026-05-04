// SR-1L-POL_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-1L-POL
// Replacement: 1L   Operator set: POL
// Equation: (((x69 + ((-2.1293101 - (x103 * 0.78548)) + x22)) - ((x65 - x111) + x28)) + x123) - x16

#include "SR-1L-POL_symbolic_head.h"
#include <math.h>

inline float _square(float v) { return v * v; }
inline float _relu(float v)   { return (v > 0.0f) ? v : 0.0f; }

// symbolic_head: pure combinational function (no loops).
// ARRAY_PARTITION complete makes every element available in the same clock
// cycle, which is required when the expression references many inputs at once.
// For dim > 128 (2L experiments) the tool may warn about register pressure;
// lower the partition factor or switch to ap_fifo streaming if needed.
float symbolic_head(float x[128]) {
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
    return (((x[69] + ((-2.1293101 - (x[103] * 0.78548)) + x[22])) - ((x[65] - x[111]) + x[28])) + x[123]) - x[16];
}
