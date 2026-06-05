#include "hybrid_mlp_1l_pol.h"

static data_t relu(acc_t x) {
#pragma HLS INLINE
    return (x > 0) ? (data_t)x : (data_t)0;
}


// Approximation for 0.78548 * x103
//
// Original POL term:
//     x103 * 0.78548
//
// Shift-add approximation:
//     0.78548 ≈ 0.5 + 0.25 + 0.03125 + 0.00390625
//             = 2^-1 + 2^-2 + 2^-5 + 2^-8
//
// This avoids a general multiplier in the symbolic head.

static acc_t approx_078548(data_t v) {
#pragma HLS INLINE

    acc_t s = 0;
    s += (acc_t)(v >> 1);
    s += (acc_t)(v >> 2);
    s += (acc_t)(v >> 5);
    s += (acc_t)(v >> 8);

    return s;
}


// SR-1L-POL symbolic head
//
// Equation:
// (((x69 + ((-2.1293101 - (x103 * 0.78548)) + x22))
//   - ((x65 - x111) + x28)) + x123) - x16
//
// Simplified grouping:
// positives: x69 + x22 + x111 + x123
// negatives: 0.78548*x103 + x65 + x28 + x16
// bias:      -2.1293101
//
// output = positives - negatives + bias

static out_t symbolic_head_pol(const data_t h2[FC2_DIM]) {
#pragma HLS INLINE

    data_t x103 = h2[103];
    data_t x22  = h2[22];
    data_t x69  = h2[69];
    data_t x65  = h2[65];
    data_t x111 = h2[111];
    data_t x28  = h2[28];
    data_t x123 = h2[123];
    data_t x16  = h2[16];

    const acc_t BIAS = (acc_t)(-2.1293101f);

    acc_t term_x103 = approx_078548(x103);

    acc_t p_x69  = (acc_t)x69;
    acc_t p_x22  = (acc_t)x22;
    acc_t p_x111 = (acc_t)x111;
    acc_t p_x123 = (acc_t)x123;

    acc_t n_x103 = term_x103;
    acc_t n_x65  = (acc_t)x65;
    acc_t n_x28  = (acc_t)x28;
    acc_t n_x16  = (acc_t)x16;

    // Balanced tree
    acc_t pos_l1_a = p_x69 + p_x22;
    acc_t pos_l1_b = p_x111 + p_x123;
    acc_t neg_l1_a = n_x103 + n_x65;
    acc_t neg_l1_b = n_x28 + n_x16;

    acc_t pos_sum = pos_l1_a + pos_l1_b;
    acc_t neg_sum = neg_l1_a + neg_l1_b;

    acc_t result = pos_sum - neg_sum + BIAS;

    return (out_t)result;
}

// ------------------------------------------------------------
// FC1: 3072 -> 512
// ------------------------------------------------------------
static void fc1_layer(
    const data_t   input[INPUT_DIM],
    const weight_t fc1_w[FC1_DIM][INPUT_DIM],
    const bias_t   fc1_b[FC1_DIM],
    data_t         h1[FC1_DIM]
) {
FC1_OUT:
    for (int i = 0; i < FC1_DIM; i++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512

        acc_t acc = (acc_t)fc1_b[i];

    FC1_IN:
        for (int j = 0; j < INPUT_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=3072 max=3072
            acc += (acc_t)input[j] * (acc_t)fc1_w[i][j];
        }

        h1[i] = relu(acc);
    }
}


// FC2: 512 -> 128
static void fc2_layer(
    const data_t   h1[FC1_DIM],
    const weight_t fc2_w[FC2_DIM][FC1_DIM],
    const bias_t   fc2_b[FC2_DIM],
    data_t         h2[FC2_DIM]
) {
FC2_OUT:
    for (int i = 0; i < FC2_DIM; i++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128

        acc_t acc = (acc_t)fc2_b[i];

    FC2_IN:
        for (int j = 0; j < FC1_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
            acc += (acc_t)h1[j] * (acc_t)fc2_w[i][j];
        }

        h2[i] = relu(acc);
    }
}


// Top function
void hybrid_mlp_1l_pol(
    const data_t   input[INPUT_DIM],
    const weight_t fc1_w[FC1_DIM][INPUT_DIM],
    const bias_t   fc1_b[FC1_DIM],
    const weight_t fc2_w[FC2_DIM][FC1_DIM],
    const bias_t   fc2_b[FC2_DIM],
    out_t          output[1]
) {
#pragma HLS INTERFACE m_axi     port=input  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=fc1_w  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=fc1_b  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=fc2_w  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi     port=fc2_b  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi     port=output offset=slave bundle=gmem5

#pragma HLS INTERFACE s_axilite port=input  bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_w  bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_b  bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_w  bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_b  bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t h1[FC1_DIM];
    data_t h2[FC2_DIM];

#pragma HLS BIND_STORAGE variable=h1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=h2 type=ram_2p impl=bram

    fc1_layer(input, fc1_w, fc1_b, h1);
    fc2_layer(h1, fc2_w, fc2_b, h2);

    output[0] = symbolic_head_pol(h2);
}
