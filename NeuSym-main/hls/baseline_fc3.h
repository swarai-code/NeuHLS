#ifndef BASELINE_FC3_H
#define BASELINE_FC3_H

#include <ap_fixed.h>

// Same precision as symbolic experiments 
// Must match feat_t / acc_t / out_t in SR-1L-* headers exactly.
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  feat_t;
typedef ap_fixed<24, 10, AP_RND, AP_SAT> acc_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  out_t;

#define FC3_IN_DIM 128

out_t baseline_fc3(feat_t x[FC3_IN_DIM], feat_t w[FC3_IN_DIM], acc_t b);

#endif // BASELINE_FC3_H
