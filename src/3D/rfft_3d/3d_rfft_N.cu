#include <mma.h>

#include "../create_fft_pfa_plan.cu"
#include "../elementwise_mul.cuh"

#include <cooperative_groups.h>

#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 4

#define WARP_SIZE 32

using namespace nvcuda;

template <int N_WARP>
__global__ void rfft_3d_8_nwarp(const double *__restrict__ input,

                                const int ACTUAL_WIDTH, const int INPUT_WIDTH, const int sub_input_width,
                                const int OVERLAP_SIZE,
                                // const int fft_n_max,

                                double *output)
{
    extern __shared__ double sharedmem[];

    double *real_shared = &sharedmem[0];
    double *imag_shared = &sharedmem[rfft_size];

    const int intput_idx_x = blockIdx.x * unit;
    const int intput_idx_y = blockIdx.y * unit;
    const int intput_idx_z = blockIdx.z * unit * 2;

    const int input_idx_real = intput_idx_x * ACTUAL_WIDTH * ACTUAL_WIDTH + intput_idx_y * ACTUAL_WIDTH + intput_idx_z;
    const int input_idx_imag = input_idx_real + unit;

    const int laneId = threadIdx.x & 0x1f;
    for (size_t i = laneId; i < rfft_size; i += WARP_SIZE)
    {
        const int x = i / (unit * unit);
        const int y = (i % (unit * unit)) / unit;
        const int z = i % unit;

        real_shared[x * (unit * unit) + y * unit + z] = input[input_idx_real + x * (ACTUAL_WIDTH * ACTUAL_WIDTH) + y * ACTUAL_WIDTH + z];
        imag_shared[x * (unit * unit) + y * unit + z] = input[input_idx_imag + x * (ACTUAL_WIDTH * ACTUAL_WIDTH) + y * ACTUAL_WIDTH + z];
    }

    cooperative_groups::thread_block g = cooperative_groups::this_thread_block();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> dft_frag_real_1[2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> dft_frag_imag_1[2];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag_real[2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag_imag[2];

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag_real[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag_imag[2];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> fft_real;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> fft_imag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> kf_real;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> kf_imag;

    wmma::load_matrix_sync(dft_frag_real_1[0], dft_matrix_real_1, unit); // a
    wmma::load_matrix_sync(dft_frag_imag_1[0], dft_matrix_imag_1, unit); // b
    wmma::load_matrix_sync(dft_frag_real_1[1], dft_matrix_real_1 + WMMA_K, unit);
    wmma::load_matrix_sync(dft_frag_imag_1[1], dft_matrix_imag_1 + WMMA_K, unit);

#pragma unroll
    for (size_t i = 0; i < unit; i++)
    {
        double *p_real = real_shared + i * WMMA_N;
        double *p_imag = imag_shared + i * WMMA_N;

        wmma::fill_fragment(fft_real, 0.0);
        wmma::fill_fragment(fft_imag, 0.0);
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            wmma::load_matrix_sync(b_frag_imag[j], p_imag + j * WMMA_K * (unit * unit), (unit * unit)); // d

            wmma::mma_sync(fft_real, dft_frag_imag_1[j], b_frag_imag[j], fft_real); // bd
            wmma::mma_sync(fft_imag, dft_frag_real_1[j], b_frag_imag[j], fft_imag); // ad

            wmma::load_matrix_sync(b_frag_real[j], p_real + j * WMMA_K * (unit * unit), (unit * unit)); // c
            wmma::mma_sync(fft_imag, dft_frag_imag_1[j], b_frag_real[j], fft_imag);                     // bc
        }
#pragma unroll
        for (int j = 0; j < fft_real.num_elements; j++)
        {
            fft_real.x[j] = -fft_real.x[j];
        }
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            wmma::mma_sync(fft_real, dft_frag_real_1[j], b_frag_real[j], fft_real); // ac
        }

        wmma::store_matrix_sync(p_real, fft_real, (unit * unit), wmma::mem_row_major);
        wmma::store_matrix_sync(p_imag, fft_imag, (unit * unit), wmma::mem_row_major);
    }

#pragma unroll
    for (size_t i = 0; i < unit; i++)
    {
        double *p_date_real = real_shared + i * (unit * unit);
        double *p_date_imag = imag_shared + i * (unit * unit);

        // (1)
        wmma::fill_fragment(kf_real, 0.0);
        wmma::fill_fragment(kf_imag, 0.0);
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            wmma::load_matrix_sync(b_frag_imag[j], p_date_imag + j * WMMA_K * unit, unit); // d

            wmma::mma_sync(kf_real, dft_frag_imag_1[j], b_frag_imag[j], kf_real); // bd
            wmma::mma_sync(kf_imag, dft_frag_real_1[j], b_frag_imag[j], kf_imag); // ad

            wmma::load_matrix_sync(b_frag_real[j], p_date_real + j * WMMA_K * unit, unit); // c
            wmma::mma_sync(kf_imag, dft_frag_imag_1[j], b_frag_real[j], kf_imag);          // bc
        }
#pragma unroll
        for (int j = 0; j < kf_real.num_elements; j++)
        {
            kf_real.x[j] = -kf_real.x[j];
        }
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            wmma::mma_sync(kf_real, dft_frag_real_1[j], b_frag_real[j], kf_real); // ac
        }

        // (2)
        wmma::fill_fragment(fft_real, 0.0);
        wmma::fill_fragment(fft_imag, 0.0);
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            wmma::load_matrix_sync(b_frag_imag[j], dft_matrix_imag_3 + j * WMMA_K * unit, unit);

            a_frag_imag[j].x[0] = kf_imag.x[j];

            wmma::mma_sync(fft_real, a_frag_imag[j], b_frag_imag[j], fft_real); // bd

            a_frag_real[j].x[0] = kf_real.x[j];

            wmma::mma_sync(fft_imag, a_frag_real[j], b_frag_imag[j], fft_imag); // ad

            wmma::load_matrix_sync(b_frag_real[j], dft_matrix_real_3 + j * WMMA_K * unit, unit);

            wmma::mma_sync(fft_imag, a_frag_imag[j], b_frag_real[j], fft_imag); // bc
        }
#pragma unroll
        for (int j = 0; j < fft_real.num_elements; j++)
        {
            fft_real.x[j] = -fft_real.x[j];
        }
        for (size_t j = 0; j < 2; j++)
        {
            wmma::mma_sync(fft_real, a_frag_real[j], b_frag_real[j], fft_real);
        }

        // TODO
        // with kernel—fft : element-wise complex multiplication between fft & ifft
        wmma::load_matrix_sync(kf_real, kernel_fft_real + i * (unit * unit), unit, wmma::mem_row_major);
        wmma::load_matrix_sync(kf_imag, kernel_fft_imag + i * (unit * unit), unit, wmma::mem_row_major);

#pragma unroll
        for (int j = 0; j < 2; j++)
        {
            complexMul(fft_real.x[j], fft_imag.x[j], kf_real.x[j], kf_imag.x[j], &fft_real.x[j], &fft_imag.x[j]);
        }

        // (3)
        wmma::fill_fragment(kf_real, 0.0);
        wmma::fill_fragment(kf_imag, 0.0);
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            b_frag_imag[j].x[0] = -b_frag_imag[j].x[0]; // d

            a_frag_imag[j].x[0] = fft_imag.x[j];                              // b
            wmma::mma_sync(kf_real, a_frag_imag[j], b_frag_imag[j], kf_real); // bd
            wmma::mma_sync(kf_imag, a_frag_imag[j], b_frag_real[j], kf_imag); // bc

            a_frag_real[j].x[0] = fft_real.x[j];                              // a
            wmma::mma_sync(kf_imag, a_frag_real[j], b_frag_imag[j], kf_imag); // ad
        }
#pragma unroll
        for (int j = 0; j < kf_real.num_elements; j++)
        {
            kf_real.x[j] = -kf_real.x[j];
        }
        for (size_t j = 0; j < 2; j++)
        {
            wmma::mma_sync(kf_real, a_frag_real[j], b_frag_real[j], kf_real); // ac
        }

        wmma::store_matrix_sync(p_date_real, kf_real, unit, wmma::mem_row_major);
        wmma::store_matrix_sync(p_date_imag, kf_imag, unit, wmma::mem_row_major);

        // (4)
        wmma::fill_fragment(fft_real, 0.0);
        wmma::fill_fragment(fft_imag, 0.0);
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            dft_frag_imag_1[j].x[0] = -dft_frag_imag_1[j].x[0]; // b

            wmma::load_matrix_sync(b_frag_imag[j], p_date_imag + j * WMMA_K * unit, unit); // d

            wmma::mma_sync(fft_real, dft_frag_imag_1[j], b_frag_imag[j], fft_real); // bd
            wmma::mma_sync(fft_imag, dft_frag_real_1[j], b_frag_imag[j], fft_imag); // ad

            wmma::load_matrix_sync(b_frag_real[j], p_date_real + j * WMMA_K * unit, unit); // c
            wmma::mma_sync(fft_imag, dft_frag_imag_1[j], b_frag_real[j], fft_imag);        // bc
        }
#pragma unroll
        for (int j = 0; j < fft_real.num_elements; j++)
        {
            fft_real.x[j] = -fft_real.x[j];
        }
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            wmma::mma_sync(fft_real, dft_frag_real_1[j], b_frag_real[j], fft_real); // ac
        }

        wmma::store_matrix_sync(p_date_real, fft_real, unit, wmma::mem_row_major);
        wmma::store_matrix_sync(p_date_imag, fft_imag, unit, wmma::mem_row_major);
    }

    // 8 * 8 @ 8 * 64 = 8 * 64
    // ifft matrix 1 @ after_kernel_complex
#pragma unroll
    for (size_t i = 0; i < unit; i++)
    {
        double *p_real = real_shared + i * WMMA_N;
        double *p_imag = imag_shared + i * WMMA_N;

        wmma::fill_fragment(fft_real, 0.0);
        wmma::fill_fragment(fft_imag, 0.0);
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            wmma::load_matrix_sync(b_frag_imag[j], p_imag + j * WMMA_K * (unit * unit), (unit * unit)); // d

            wmma::mma_sync(fft_real, dft_frag_imag_1[j], b_frag_imag[j], fft_real); // bd
            wmma::mma_sync(fft_imag, dft_frag_real_1[j], b_frag_imag[j], fft_imag); // ad

            wmma::load_matrix_sync(b_frag_real[j], p_real + j * WMMA_K * (unit * unit), (unit * unit)); // c
            wmma::mma_sync(fft_imag, dft_frag_imag_1[j], b_frag_real[j], fft_imag);                     // bc
        }
#pragma unroll
        for (int j = 0; j < fft_real.num_elements; j++)
        {
            fft_real.x[j] = -fft_real.x[j];
        }
#pragma unroll
        for (size_t j = 0; j < 2; j++)
        {
            wmma::mma_sync(fft_real, dft_frag_real_1[j], b_frag_real[j], fft_real); // ac
        }

        wmma::store_matrix_sync(p_real, fft_real, (unit * unit), wmma::mem_row_major);
        wmma::store_matrix_sync(p_imag, fft_imag, (unit * unit), wmma::mem_row_major);

        // wmma::store_matrix_sync(output + i * WMMA_N, fft_real, (unit * unit), wmma::mem_row_major);
    }

    const int output_idx_x = blockIdx.x * sub_input_width;
    const int output_idx_y = blockIdx.y * sub_input_width;
    const int output_idx_z = blockIdx.z * sub_input_width * 2;

    for (size_t i = laneId; i < rfft_size; i += WARP_SIZE)
    {
        const int x = i / (unit * unit);
        const int y = (i % (unit * unit)) / unit;
        const int z = i % unit;

        int idx_x = (output_idx_x + x);
        int idx_y = (output_idx_y + y);
        int idx_z_real = (output_idx_z + z);
        int idx_z_imag = (output_idx_z + z + sub_input_width);

        if (blockIdx.x == gridDim.x - 1)
        {
            idx_x = (x >= sub_input_width) ? (x - sub_input_width) : (output_idx_x + x);
        }
        if (blockIdx.y == gridDim.y - 1)
        {
            idx_y = (y >= sub_input_width) ? (y - sub_input_width) : (output_idx_y + y);
        }
        if (blockIdx.z == gridDim.z - 1)
        {
            idx_z_imag = (z >= sub_input_width) ? (z - sub_input_width) : (output_idx_z + z + sub_input_width);
        }

        atomicAdd(output + idx_x * INPUT_WIDTH * INPUT_WIDTH + idx_y * INPUT_WIDTH + idx_z_real, real_shared[i]);
        atomicAdd(output + idx_x * INPUT_WIDTH * INPUT_WIDTH + idx_y * INPUT_WIDTH + idx_z_imag, imag_shared[i]);
    }
}