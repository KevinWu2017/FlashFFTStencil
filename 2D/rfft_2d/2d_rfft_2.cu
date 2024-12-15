#include <mma.h>

#include "../create_fft_pfa_plan.cu"
#include "../elementwise_mul.cuh"

#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 4

#define WARP_SIZE 32

using namespace nvcuda;

template <int N_WARP>
__global__ void rfft_2d_8_nwarp(const double *__restrict__ input,

                                const int ACTUAL_WIDTH,

                                const int INPUT_WIDTH,

                                const int sub_input_width,

                                const int OVERLAP_SIZE,

                                double *output)
{
    extern __shared__ double sharedmem[];

    const int warp_id = threadIdx.x / WARP_SIZE; // TODO: 1
    double *real_shared = &sharedmem[0 + warp_id * rfft_size];
    double *imag_shared = &sharedmem[rfft_size * N_WARP + warp_id * rfft_size];

    const int laneId = threadIdx.x & 0x1f;

    // for (size_t i = laneId; i < rfft_size; i += WARP_SIZE)
    // {
    //     const int row = i / unit;
    //     const int col = i % unit;

    //     real_shared[i] = input[input_idx_real + row * ACTUAL_WIDTH + col];
    //     imag_shared[i] = input[input_idx_imag + row * ACTUAL_WIDTH + col];
    // }
    //

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> dft_frag_real_1[2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> dft_frag_imag_1[2];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag_real[2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag_imag[2];

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag_real[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag_imag[2];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> fft_real;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> fft_imag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> k_real;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> k_imag;

    wmma::load_matrix_sync(dft_frag_real_1[0], dft_matrix_real_1, unit); // a
    wmma::load_matrix_sync(dft_frag_imag_1[0], dft_matrix_imag_1, unit); // b
    wmma::load_matrix_sync(dft_frag_real_1[1], dft_matrix_real_1 + WMMA_K, unit);
    wmma::load_matrix_sync(dft_frag_imag_1[1], dft_matrix_imag_1 + WMMA_K, unit);

    const int intput_idx_x = blockIdx.y * unit;
    const int intput_idx_y = blockIdx.x * 2 * unit;
    const int input_idx_real = intput_idx_x * ACTUAL_WIDTH + intput_idx_y;
    const int input_idx_imag = input_idx_real + unit;

    /**
     * @brief f f t
     *
     */
    // 8 * 8 @ 8 * 8（7）
    // dht_matrix_row @ real_shared
    wmma::fill_fragment(k_real, 0.0);
    wmma::fill_fragment(k_imag, 0.0);

#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        wmma::load_matrix_sync(b_frag_imag[i], input + input_idx_imag + i * WMMA_K * ACTUAL_WIDTH, ACTUAL_WIDTH); // d

        wmma::mma_sync(k_real, dft_frag_imag_1[i], b_frag_imag[i], k_real); // bd
        wmma::mma_sync(k_imag, dft_frag_real_1[i], b_frag_imag[i], k_imag); // ad

        wmma::load_matrix_sync(b_frag_real[i], input + input_idx_real + i * WMMA_K * ACTUAL_WIDTH, ACTUAL_WIDTH); // c
        wmma::mma_sync(k_imag, dft_frag_imag_1[i], b_frag_real[i], k_imag);                                       // bc
    }
#pragma unroll
    for (int i = 0; i < k_real.num_elements; i++)
    {
        k_real.x[i] = -k_real.x[i];
    }
#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        wmma::mma_sync(k_real, dft_frag_real_1[i], b_frag_real[i], k_real); // ac
    }

    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    wmma::fill_fragment(fft_real, 0.0);
    wmma::fill_fragment(fft_imag, 0.0);
    // real_shared @ dht_matrix_col (8 * 8 @  8 * 8)
#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        wmma::load_matrix_sync(b_frag_imag[i], dft_matrix_imag_2 + i * WMMA_K * unit, unit); // d

        a_frag_imag[i].x[0] = k_imag.x[i];                                  // b
        wmma::mma_sync(fft_real, a_frag_imag[i], b_frag_imag[i], fft_real); // bd

        a_frag_real[i].x[0] = k_real.x[i];                                  // a
        wmma::mma_sync(fft_imag, a_frag_real[i], b_frag_imag[i], fft_imag); // ad

        wmma::load_matrix_sync(b_frag_real[i], dft_matrix_real_2 + i * WMMA_K * unit, unit); // c
        wmma::mma_sync(fft_imag, a_frag_real[i], b_frag_real[i], fft_imag);                  // bc
    }
#pragma unroll
    for (int i = 0; i < fft_real.num_elements; i++)
    {
        fft_real.x[i] = -fft_real.x[i];
    }
#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        wmma::mma_sync(fft_real, a_frag_real[i], b_frag_real[i], fft_real); // ac
    }

    // --------------------------------------------------------------------------------------------------------------
    // -------------------------------kernel multiplication----------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    // element-wise complex multiplication between fft & ifft
    wmma::load_matrix_sync(k_real, kernel_fft_real, unit, wmma::mem_row_major);
    wmma::load_matrix_sync(k_imag, kernel_fft_imag, unit, wmma::mem_row_major);
#pragma unroll
    for (int i = 0; i < k_real.num_elements; i++)
    {
        complexMul(fft_real.x[i], fft_imag.x[i], k_real.x[i], k_imag.x[i], &k_real.x[i], &k_imag.x[i]);
    }

    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    /**
     * @brief i f h t
     *
     */
    // real_shared @ idht_matrix_col, 8 * 8（7） @  8 * 8
    wmma::fill_fragment(fft_real, 0.0);
    wmma::fill_fragment(fft_imag, 0.0);

#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        b_frag_imag[i].x[0] = -b_frag_imag[i].x[0]; // d

        a_frag_imag[i].x[0] = k_imag.x[i];                                  // b
        wmma::mma_sync(fft_real, a_frag_imag[i], b_frag_imag[i], fft_real); // bd
        wmma::mma_sync(fft_imag, a_frag_imag[i], b_frag_real[i], fft_imag); // bc

        a_frag_real[i].x[0] = k_real.x[i];                                  // a
        wmma::mma_sync(fft_imag, a_frag_real[i], b_frag_imag[i], fft_imag); // ad
    }

#pragma unroll
    for (int i = 0; i < fft_real.num_elements; i++)
    {
        fft_real.x[i] = -fft_real.x[i];
    }

#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        wmma::mma_sync(fft_real, a_frag_real[i], b_frag_real[i], fft_real); // ac
    }

    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    wmma::store_matrix_sync(real_shared, fft_real, unit, wmma::mem_row_major);
    wmma::store_matrix_sync(imag_shared, fft_imag, unit, wmma::mem_row_major);

    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    wmma::fill_fragment(fft_real, 0.0);
    wmma::fill_fragment(fft_imag, 0.0);

    // 8 * 8 @ 8 * 8（7）
    // idht_matrix_row @ real_shared
#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {

        dft_frag_imag_1[i].x[0] = -dft_frag_imag_1[i].x[0]; // b

        wmma::load_matrix_sync(b_frag_imag[i], imag_shared + i * WMMA_K * unit, unit); // d

        wmma::mma_sync(fft_real, dft_frag_imag_1[i], b_frag_imag[i], fft_real); // bd
        wmma::mma_sync(fft_imag, dft_frag_real_1[i], b_frag_imag[i], fft_imag); // ad

        wmma::load_matrix_sync(b_frag_real[i], real_shared + i * WMMA_K * unit, unit); // c
        wmma::mma_sync(fft_imag, dft_frag_imag_1[i], b_frag_real[i], fft_imag);        // bc
    }

#pragma unroll
    for (int i = 0; i < fft_real.num_elements; i++)
    {
        fft_real.x[i] = -fft_real.x[i];
    }

#pragma unroll
    for (size_t i = 0; i < 2; i++)
    {
        wmma::mma_sync(fft_real, dft_frag_real_1[i], b_frag_real[i], fft_real); // ac
    }

    // wmma::store_matrix_sync(real_shared, fft_real, unit, wmma::mem_row_major);
    // wmma::store_matrix_sync(imag_shared, fft_imag, unit, wmma::mem_row_major);

    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------------

    // const int fft_id_x = blockIdx.x;
    // const int fft_id_y = blockIdx.y;

    const int output_idx_x = blockIdx.y * sub_input_width;
    const int output_idx_y = blockIdx.x * 2 * sub_input_width;

    // if ((fft_id_x != gridDim.x - 1) && (fft_id_y != gridDim.y - 1))
    // {
        for (size_t i = laneId; i < rfft_size; i += WARP_SIZE)
        {
            const int row = i / unit;
            const int col = i % unit;

            const int idx_x = (output_idx_x + row);
            const int idx_y_real = (output_idx_y + col);
            const int idx_y_imag = (output_idx_y + sub_input_width + col);

            atomicAdd(output + idx_x * INPUT_WIDTH + idx_y_real, real_shared[i]);
            atomicAdd(output + idx_x * INPUT_WIDTH + idx_y_imag, imag_shared[i]);
        }
    // }
    // else if ((fft_id_x == gridDim.x - 1) && (fft_id_y != gridDim.y - 1))
    // {
    //     for (size_t i = laneId; i < rfft_size; i += WARP_SIZE)
    //     {
    //         const int row = i / unit;
    //         const int col = i % unit;

    //         const int idx_x = (output_idx_x + row);
    //         const int idx_y_real = (output_idx_y + col);
    //         const int idx_y_imag =
    //             (col >= sub_input_width) ? (col - sub_input_width) : (output_idx_y + sub_input_width + col);

    //         atomicAdd(output + idx_x * INPUT_WIDTH + idx_y_real, real_shared[i]);
    //         atomicAdd(output + idx_x * INPUT_WIDTH + idx_y_imag, imag_shared[i]);
    //     }
    // }
    // else if ((fft_id_x != gridDim.x - 1) && (fft_id_y == gridDim.y - 1))
    // {
    //     for (size_t i = laneId; i < rfft_size; i += WARP_SIZE)
    //     {
    //         const int row = i / unit;
    //         const int col = i % unit;

    //         const int idx_x = (row >= sub_input_width) ? (row - sub_input_width) : (output_idx_x + row);
    //         const int idx_y_real = (output_idx_y + col);
    //         const int idx_y_imag = (output_idx_y + sub_input_width + col);

    //         atomicAdd(output + idx_x * INPUT_WIDTH + idx_y_real, real_shared[i]);
    //         atomicAdd(output + idx_x * INPUT_WIDTH + idx_y_imag, imag_shared[i]);
    //     }
    // }
    // else if ((fft_id_x == gridDim.x - 1) && (fft_id_y == gridDim.y - 1))
    // {
    //     for (size_t i = laneId; i < rfft_size; i += WARP_SIZE)
    //     {
    //         const int row = i / unit;
    //         const int col = i % unit;

    //         const int idx_x = (row >= sub_input_width) ? (row - sub_input_width) : (output_idx_x + row);
    //         const int idx_y_real = (output_idx_y + col);
    //         const int idx_y_imag =
    //             (col >= sub_input_width) ? (col - sub_input_width) : (output_idx_y + sub_input_width + col);

    //         atomicAdd(output + idx_x * INPUT_WIDTH + idx_y_real, real_shared[i]);
    //         atomicAdd(output + idx_x * INPUT_WIDTH + idx_y_imag, imag_shared[i]);
    //     }
    // }
}