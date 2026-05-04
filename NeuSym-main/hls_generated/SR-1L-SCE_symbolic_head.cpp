// SR-1L-SCE_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-1L-SCE
// Replacement: 1L   Operator set: SCE
// Equation: (((x121 + ((x24 - x0) + (x63 - x47))) + (-4.668614 - x66)) * 0.7971458) - (x38 * 2.8560774)

#include "SR-1L-SCE_symbolic_head.h"
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
    return (((x[121] + ((x[24] - x[0]) + (x[63] - x[47]))) + (-4.668614 - x[66])) * 0.7971458) - (x[38] * 2.8560774);
}
