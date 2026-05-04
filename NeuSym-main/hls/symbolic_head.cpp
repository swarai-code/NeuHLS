// hls/symbolic_head.cpp
// Generic stub — replace with a generated file from outputs/hls/
// after running: python generate_hls.py --all
//
// Generated files use a sparse scalar parameter interface:
//   float symbolic_head(float x_3, float x_17, float x_42) { return ...; }
// Only variables that appear in the equation are declared as parameters.
//
// Why scalar params instead of float x[N]:
//   - Eliminates the input mux (saved ~366 LUTs for a 128-input array)
//   - No ARRAY_PARTITION needed — scalars already have individual ports
//   - Applies to both 1L (128-dim input) and 2L (512-dim input) experiments
//   - The equation is combinatorial (no loops), so array partitioning
//     would only add register pressure without improving throughput

#include "symbolic_head.h"
#include <math.h>

inline float _square(float v) { return v * v; }
inline float _relu(float v)   { return (v > 0.0f) ? v : 0.0f; }

// Placeholder — copy the experiment-specific file from outputs/hls/ here
// and update the #include above to match its header.
float symbolic_head(float x_0) {
    return x_0;
}
