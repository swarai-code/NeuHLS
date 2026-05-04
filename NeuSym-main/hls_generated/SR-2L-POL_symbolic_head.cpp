// SR-2L-POL_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-2L-POL
// Replacement: 2L   Operator set: POL
// Equation: (x347 - (((x460 + (x107 + x306)) - x245) - 0.7044616)) / 0.95166284

#include "SR-2L-POL_symbolic_head.h"
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
    return (x[347] - (((x[460] + (x[107] + x[306])) - x[245]) - 0.7044616)) / 0.95166284;
}
