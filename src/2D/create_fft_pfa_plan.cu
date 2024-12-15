#include "./helper_cuda/helper_cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cufft.h>
#include <iostream>

__constant__ double dft_matrix_real_1[rfft_size];
__constant__ double dft_matrix_imag_1[rfft_size];

__constant__ double dft_matrix_real_2[rfft_size];
__constant__ double dft_matrix_imag_2[rfft_size];

__constant__ double kernel_fft_real[rfft_size];
__constant__ double kernel_fft_imag[rfft_size];

void CreatePlan(double *k, const int KERNEL_WIDTH, bool print_matrix)
{
    double *host_dft_matrix_real_1 = (double *)malloc(rfft_size * sizeof(double));
    double *host_dft_matrix_imag_1 = (double *)malloc(rfft_size * sizeof(double));

    double *host_dft_matrix_real_2 = (double *)malloc(rfft_size * sizeof(double));
    double *host_dft_matrix_imag_2 = (double *)malloc(rfft_size * sizeof(double));

    for (int i = 0; i < unit; ++i)
    {
        int yushu = i % 8;
        int beishu = i / 8;
        int row_index = beishu * 8 + yushu % 2 * 4 + yushu / 2;

        for (int j = 0; j < unit; ++j)
        {
            host_dft_matrix_real_1[unit * i + j] = cos(2 * M_PI * i * j / unit);
            host_dft_matrix_imag_1[unit * i + j] = -sin(2 * M_PI * i * j / unit);

            host_dft_matrix_real_2[unit * row_index + j] = cos(2 * M_PI * i * j / unit);
            host_dft_matrix_imag_2[unit * row_index + j] = -sin(2 * M_PI * i * j / unit);
        }
    }

    if (print_matrix)
    {
        printf("DFT unit Matrix 1\n");
        for (int i = 0; i < unit; ++i)
        {

            for (int j = 0; j < unit; ++j)
            {
                printf("(%f, %f)\t", host_dft_matrix_real_1[unit * i + j], host_dft_matrix_imag_1[unit * i + j]);
            }
            printf("\n");
        }
        std::cout << std::endl;

        printf("DFT unit Matrix 2\n");
        for (int i = 0; i < unit; ++i)
        {

            for (int j = 0; j < unit; ++j)
            {
                printf("(%f, %f)\t", host_dft_matrix_real_2[unit * i + j], host_dft_matrix_imag_2[unit * i + j]);
            }
            printf("\n");
        }
        std::cout << std::endl;
    }

    // cuda malloc
    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_real_1, host_dft_matrix_real_1, rfft_size * sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_imag_1, host_dft_matrix_imag_1, rfft_size * sizeof(double)));

    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_real_2, host_dft_matrix_real_2, rfft_size * sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_imag_2, host_dft_matrix_imag_2, rfft_size * sizeof(double)));

    // part 2: create FFT kernel
    double2 *h_kernel = (double2 *)calloc(unit * unit, sizeof(double2));
    double2 *h_fft_kernel = (double2 *)calloc(unit * unit, sizeof(double2));
    for (size_t i = 0; i < KERNEL_WIDTH; i++)
    {
        for (size_t j = 0; j < KERNEL_WIDTH; j++)
        {
            h_kernel[i * unit + j] = make_double2(k[i * KERNEL_WIDTH + j], 0.0);
        }
    }
    double2 *d_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_kernel, unit * unit * sizeof(double2)));
    double2 *d_fft_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_fft_kernel, unit * unit * sizeof(double2)));

    checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, unit * unit * sizeof(double2), cudaMemcpyHostToDevice));

    cufftHandle plan_2d_fft;
    checkCudaErrors(cufftPlan2d(&plan_2d_fft, unit, unit, CUFFT_Z2Z));
    checkCudaErrors(
        cufftExecZ2Z(plan_2d_fft, (cufftDoubleComplex *)d_kernel, (cufftDoubleComplex *)d_fft_kernel, CUFFT_FORWARD));

    checkCudaErrors(cudaMemcpy(h_fft_kernel, d_fft_kernel, unit * unit * sizeof(double2), cudaMemcpyDeviceToHost));

    double *h_kernel_fft_real = (double *)malloc(rfft_size * sizeof(double));
    double *h_kernel_fft_imag = (double *)malloc(rfft_size * sizeof(double));
    for (int i = 0; i < (unit * unit); ++i)
    {
        h_kernel_fft_real[i] = h_fft_kernel[i].x / (unit * unit);
        h_kernel_fft_imag[i] = h_fft_kernel[i].y / (unit * unit);
        // h_kernel_fft_real[i] = h_fft_kernel[i].x ;
        // h_kernel_fft_imag[i] = h_fft_kernel[i].y ;
    }

    if (print_matrix)
    {

        printf("Kernel Matrix\n");
        for (int i = 0; i < unit; ++i)
        {
            for (int j = 0; j < unit; ++j)
            {
                printf("(%f, %f) \t", h_kernel[unit * i + j].x, h_kernel[unit * i + j].y);
            }
            printf("\n");
        }

        printf("Kernel_FFT Matrix\n");
        for (int i = 0; i < unit; ++i)
        {
            for (int j = 0; j < unit; ++j)
            {
                printf("(%f, %f) \t", h_kernel_fft_real[unit * i + j], h_kernel_fft_imag[unit * i + j]);
            }
            printf("\n");
        }
        std::cout << std::endl;
    }

    checkCudaErrors(cudaMemcpyToSymbol(kernel_fft_real, h_kernel_fft_real, rfft_size * sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(kernel_fft_imag, h_kernel_fft_imag, rfft_size * sizeof(double)));

    // free(host_dft_matrix_real_1);
    // free(host_dft_matrix_imag_1);

    // free(host_dft_matrix_real_2);
    // free(host_dft_matrix_real_2);

    // free(host_idft_matrix_real_1);
    // free(host_idft_matrix_imag_1);

    // free(host_idft_matrix_real_2);
    // free(host_idft_matrix_imag_2);
}