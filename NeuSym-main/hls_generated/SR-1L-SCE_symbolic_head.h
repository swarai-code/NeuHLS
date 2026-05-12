#ifndef SR_1L_SCE_SYMBOLIC_HEAD_H
#define SR_1L_SCE_SYMBOLIC_HEAD_H

#include <ap_fixed.h>

// Precision config
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  feat_t;   // input feature
typedef ap_fixed<24, 10, AP_RND, AP_SAT> acc_t;    // accumulator (wider to avoid overflow)
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  out_t;    // output

// Sparsity-aware interface
// Only the 7 active features are passed in — NOT the full x[128].
// Without ARRAY_PARTITION, sparse_x is BRAM-backed (ap_memory interface).
// All inputs are read into local scalars at the top of symbolic_head()
// before any arithmetic begins, giving the scheduler full visibility
// of the access pattern under BRAM port constraints.
//
// Caller packs before invoking:
//   sparse_x[0] = x[121]
//   sparse_x[1] = x[24]
//   sparse_x[2] = x[0]
//   sparse_x[3] = x[63]
//   sparse_x[4] = x[47]
//   sparse_x[5] = x[66]
//   sparse_x[6] = x[38]
//
// If II=1 is required, consider flattening to 7 scalar ports:
//   out_t symbolic_head(feat_t x121, feat_t x24, feat_t x0, feat_t x63,
//                       feat_t x47, feat_t x66, feat_t x38);
// This avoids BRAM entirely with no pragmas needed.

#define SYMBOLIC_NUM_TERMS 7

out_t symbolic_head(feat_t sparse_x[SYMBOLIC_NUM_TERMS]);

#endif // SR_1L_SCE_SYMBOLIC_HEAD_H
