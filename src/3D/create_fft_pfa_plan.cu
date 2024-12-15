#include "./helper_cuda/helper_cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cufft.h>
#include <iostream>

__constant__ double dft_matrix_real_1[(unit * unit)];
__constant__ double dft_matrix_imag_1[(unit * unit)];

__constant__ double dft_matrix_real_3[(unit * unit)];
__constant__ double dft_matrix_imag_3[(unit * unit)];

__constant__ double kernel_fft_real[rfft_size];
__constant__ double kernel_fft_imag[rfft_size];

void CreatePlan(double *kernel, const int KERNEL_WIDTH, bool print_matrix)
{
    double *host_dft_matrix_real_1 = (double *)malloc((unit * unit) * sizeof(double));
    double *host_dft_matrix_imag_1 = (double *)malloc((unit * unit) * sizeof(double));

    double *host_dft_matrix_real_3 = (double *)malloc((unit * unit) * sizeof(double));
    double *host_dft_matrix_imag_3 = (double *)malloc((unit * unit) * sizeof(double));

    for (int i = 0; i < unit; ++i)
    {
        for (int j = 0; j < unit; ++j)
        {
            int yushu = i % 8;
            int beishu = i / 8;
            int row_index = beishu * 8 + yushu % 2 * 4 + yushu / 2;

            host_dft_matrix_real_1[unit * i + j] = cos(2 * M_PI * i * j / unit);
            host_dft_matrix_imag_1[unit * i + j] = -sin(2 * M_PI * i * j / unit);

            host_dft_matrix_real_3[unit * row_index + j] = cos(2 * M_PI * i * j / unit);
            host_dft_matrix_imag_3[unit * row_index + j] = -sin(2 * M_PI * i * j / unit);
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
    }

    // cuda malloc
    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_real_1, host_dft_matrix_real_1, (unit * unit) * sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_imag_1, host_dft_matrix_imag_1, (unit * unit) * sizeof(double)));

    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_real_3, host_dft_matrix_real_3, (unit * unit) * sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(dft_matrix_imag_3, host_dft_matrix_imag_3, (unit * unit) * sizeof(double)));

    // part 2: create FFT kernel
    double2 *h_kernel = (double2 *)calloc(rfft_size, sizeof(double2));
    double2 *h_fft_kernel = (double2 *)calloc(rfft_size, sizeof(double2));
    for (size_t i = 0; i < KERNEL_WIDTH; i++)
    {
        for (size_t j = 0; j < KERNEL_WIDTH; j++)
        {
            for (size_t k = 0; k < KERNEL_WIDTH; k++)
            {
                h_kernel[i * unit * unit + j * unit + k] =
                    make_double2(kernel[i * KERNEL_WIDTH * KERNEL_WIDTH + j * KERNEL_WIDTH + k], 0.0);
            }
        }
    }
    double2 *d_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_kernel, rfft_size * sizeof(double2)));
    double2 *d_fft_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_fft_kernel, rfft_size * sizeof(double2)));

    checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, rfft_size * sizeof(double2), cudaMemcpyHostToDevice));

    cufftHandle plan_3d_fft;
    checkCudaErrors(cufftPlan3d(&plan_3d_fft, unit, unit, unit, CUFFT_Z2Z));
    checkCudaErrors(
        cufftExecZ2Z(plan_3d_fft, (cufftDoubleComplex *)d_kernel, (cufftDoubleComplex *)d_fft_kernel, CUFFT_FORWARD));

    checkCudaErrors(cudaMemcpy(h_fft_kernel, d_fft_kernel, rfft_size * sizeof(double2), cudaMemcpyDeviceToHost));

    double *h_kernel_fft_real = (double *)malloc(rfft_size * sizeof(double));
    double *h_kernel_fft_imag = (double *)malloc(rfft_size * sizeof(double));
    for (int i = 0; i < rfft_size; ++i)
    {
        h_kernel_fft_real[i] = h_fft_kernel[i].x / (rfft_size);
        h_kernel_fft_imag[i] = h_fft_kernel[i].y / (rfft_size);

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
                for (int k = 0; k < unit; ++k)
                {
                    printf("(%f, %f) \t", h_kernel[i * unit * unit + j * unit + k].x,
                           h_kernel[i * unit * unit + j * unit + k].y);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
        printf("\n");

        printf("Kernel_FFT Matrix\n");
        for (int i = 0; i < unit; ++i)
        {
            for (int j = 0; j < unit; ++j)
            {
                for (int k = 0; k < unit; ++k)
                {
                    printf("(%f, %f) \t", h_kernel_fft_real[i * unit * unit + j * unit + k],
                           h_kernel_fft_imag[i * unit * unit + j * unit + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
        std::cout << std::endl;
    }

    checkCudaErrors(cudaMemcpyToSymbol(kernel_fft_real, h_kernel_fft_real, rfft_size * sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(kernel_fft_imag, h_kernel_fft_imag, rfft_size * sizeof(double)));
}