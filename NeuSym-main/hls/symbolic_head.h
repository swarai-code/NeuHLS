// hls/symbolic_head.h
// Generic stub header — signature varies per experiment.
// Copy the matching header from outputs/hls/ for Vitis HLS synthesis.
//
// 1L experiments: h2 is 128-dim; equation uses a sparse subset of those inputs.
// 2L experiments: h1 is 512-dim; equation uses a sparse subset of those inputs.
// Both cases use scalar float parameters — no array, no ARRAY_PARTITION required.
#ifndef SYMBOLIC_HEAD_H
#define SYMBOLIC_HEAD_H

// Placeholder — replace with the exact signature from outputs/hls/*_symbolic_head.h
float symbolic_head(float x_0);

#endif  // SYMBOLIC_HEAD_H
