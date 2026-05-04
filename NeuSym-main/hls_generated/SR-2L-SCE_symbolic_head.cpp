// SR-2L-SCE_symbolic_head.cpp
// Auto-generated HLS C++ for experiment: SR-2L-SCE
// Replacement: 2L   Operator set: SCE
// Equation: (((x455 + 1.3092213) + (x164 - x306)) - (x250 + x255)) - (x399 + ((cos(x164) - x227) * -0.42439273))

#include "SR-2L-SCE_symbolic_head.h"
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
    return (((x[455] + 1.3092213) + (x[164] - x[306])) - (x[250] + x[255])) - (x[399] + ((cosf(x[164]) - x[227]) * -0.42439273));
}
