// baseline_fc2_fc3.h
// HLS header for baseline fc2+fc3: dense 512 -> 128 (ReLU) -> 1
// Comparison target for 2L symbolic experiments.

#ifndef BASELINE_FC2_FC3_H
#define BASELINE_FC2_FC3_H

#define FC2_IN_DIM  512
#define FC2_OUT_DIM 128

// Weights are passed as arguments so the testbench can load trained values.
float baseline_fc2_fc3(
    float x[FC2_IN_DIM],
    float W2[FC2_OUT_DIM][FC2_IN_DIM],
    float B2[FC2_OUT_DIM],
    float W3[FC2_OUT_DIM],   // fc3 weights: 128 -> 1
    float B3                 // fc3 bias
);

#endif  // BASELINE_FC2_FC3_H
