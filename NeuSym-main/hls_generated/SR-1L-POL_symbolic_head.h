#ifndef SR_1L_POL_SYMBOLIC_HEAD_H
#define SR_1L_POL_SYMBOLIC_HEAD_H

#include <ap_fixed.h>

// Precision config
// W=16, I=8  →  8 integer bits + 8 fractional bits (Q8.8)
// Change W/I here to trade accuracy vs. resource usage
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  feat_t;   // input feature
typedef ap_fixed<24, 10, AP_RND, AP_SAT> acc_t;    // accumulator (wider to avoid overflow)
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  out_t;    // output

// Sparsity-aware interface
// Only the 8 active features are passed in — NOT the full x[128].
// Without ARRAY_PARTITION, sparse_x is BRAM-backed (ap_memory interface).
// All inputs are read into local scalars at the top of symbolic_head()
// before any arithmetic begins, giving the scheduler full visibility
// of the access pattern under BRAM port constraints.
//
// Caller is responsible for selecting these before invoking:
//   sparse_x[0] = x[103]
//   sparse_x[1] = x[22]
//   sparse_x[2] = x[69]
//   sparse_x[3] = x[65]
//   sparse_x[4] = x[111]
//   sparse_x[5] = x[28]
//   sparse_x[6] = x[123]
//   sparse_x[7] = x[16]
//
// If II=1 is required, consider flattening to 8 scalar ports:
//   out_t symbolic_head(feat_t x103, feat_t x22, feat_t x69, feat_t x65,
//                       feat_t x111, feat_t x28, feat_t x123, feat_t x16);
// This avoids BRAM entirely with no pragmas needed.

#define SYMBOLIC_NUM_TERMS 8

out_t symbolic_head(feat_t sparse_x[SYMBOLIC_NUM_TERMS]);

#endif // SR_1L_POL_SYMBOLIC_HEAD_H
