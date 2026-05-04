// baseline_fc3.h
// HLS header for baseline fc3: dense 128 → 1
// Direct comparison target for 1L symbolic experiments.

#ifndef BASELINE_FC3_H
#define BASELINE_FC3_H

#define FC3_IN_DIM 128

float baseline_fc3(float x[FC3_IN_DIM], float w[FC3_IN_DIM], float b);

#endif  // BASELINE_FC3_H
