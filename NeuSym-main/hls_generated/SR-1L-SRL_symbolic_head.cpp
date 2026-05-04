// SR-1L-SRL_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-1L-SRL
// Replacement: 1L   Operator set: SRL
// Equation: (((x7 - x3) - ((x65 + relu(x27)) - (x24 - x118))) * 0.72398) - ((x74 * 0.21851327) - x22)

#include "SR-1L-SRL_symbolic_head.h"
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
    return (((x[7] - x[3]) - ((x[65] + _relu(x[27])) - (x[24] - x[118]))) * 0.72398) - ((x[74] * 0.21851327) - x[22]);
}
