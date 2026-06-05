#ifndef HYBRID_MLP_1L_POL_H
#define HYBRID_MLP_1L_POL_H

#include <ap_fixed.h>

// Network dimensions
#define INPUT_DIM 3072
#define FC1_DIM   512
#define FC2_DIM   128

// Fixed-point precision
// You can tune this later: 16,8 is consistent with your symbolic head.
typedef ap_fixed<16, 8, AP_RND, AP_SAT> data_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT> weight_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT> bias_t;
typedef ap_fixed<24, 10, AP_RND, AP_SAT> acc_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT> out_t;

// Top function:
// Full hybrid network:
// x -> fc1 -> ReLU -> fc2 -> ReLU -> symbolic POL head
void hybrid_mlp_1l_pol(
    const data_t   input[INPUT_DIM],
    const weight_t fc1_w[FC1_DIM][INPUT_DIM],
    const bias_t   fc1_b[FC1_DIM],
    const weight_t fc2_w[FC2_DIM][FC1_DIM],
    const bias_t   fc2_b[FC2_DIM],
    out_t          output[1]
);

#endif
