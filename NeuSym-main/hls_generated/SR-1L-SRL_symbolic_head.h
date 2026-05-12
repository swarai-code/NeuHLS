#ifndef SR_1L_SRL_SYMBOLIC_HEAD_H
#define SR_1L_SRL_SYMBOLIC_HEAD_H

#include <ap_fixed.h>

// Precision config
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  feat_t;   // input feature
typedef ap_fixed<24, 10, AP_RND, AP_SAT> acc_t;    // accumulator (wider to avoid overflow)
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  out_t;    // output

// Sparsity-aware interface
// Only the 8 active features are passed in — NOT the full x[128].
// Without ARRAY_PARTITION, sparse_x is BRAM-backed (ap_memory interface).
// HLS will schedule reads across available BRAM ports (1 or 2 per cycle).
// All inputs are read into local scalars at the top of symbolic_head()
// before any arithmetic begins, giving the scheduler full visibility.
//
// Caller packs before invoking:
//   sparse_x[0] = x[7]
//   sparse_x[1] = x[3]
//   sparse_x[2] = x[65]
//   sparse_x[3] = x[27]   ← relu applied internally
//   sparse_x[4] = x[24]
//   sparse_x[5] = x[118]
//   sparse_x[6] = x[74]
//   sparse_x[7] = x[22]
//
// If II=1 is required, consider flattening to 8 scalar ports:
//   out_t symbolic_head(feat_t x7, feat_t x3, feat_t x65, feat_t x27,
//                       feat_t x24, feat_t x118, feat_t x74, feat_t x22);
// This avoids BRAM entirely with no pragmas needed.

#define SYMBOLIC_NUM_TERMS 8

out_t symbolic_head(feat_t sparse_x[SYMBOLIC_NUM_TERMS]);

#endif // SR_1L_SRL_SYMBOLIC_HEAD_H
