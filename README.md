# FlashFFTStencil: Bridging Fast Fourier Transforms to Memory-Efficient Stencil Computations on Tensor Core Units

This repository contains the official code for FlashFFTStencil, a memory-efficient stencil computing system designed to bridge fast Fourier transforms to fully-dense stencil computations on Tensor Core Units.

FlashFFTStencil demonstrates remarkable efficiency in stencil computations, achieving an average speedup of 2.57× over the current state-of-the-art methods. Notably, in 1D cases, it achieves an exceptional 103.0× speedup compared to stencil implementations based on cuFFT.


**FlashFFTStencil: Bridging Fast Fourier Transforms to Memory-Efficient Stencil Computations on Tensor Core Units (PPoPP'25)** \
Paper: https://ppopp25.sigplan.org/track/PPoPP-2025-Main-Conference-1 \
Blog: https://mp.weixin.qq.com/s/KBUiKvvXqAHB0YC5XdQ4ww


![FlashFFTStencil](assets/intro.png)

If you have any questions or would like to discuss more detail, please feel free to reach out to Haozhi at **haozhi.han@stu.pku.edu.cn**.


## Installation

Recommended Setup:
We suggest utilizing the Nvidia PyTorch Docker container, as this library has been developed and validated using version 23.05.

* This library has been tested with CUDA 12.1 and its corresponding toolkit.
* While our testing was conducted on A100 and H100 GPUs, it should work seamlessly on any Ampere/Hopper architecture (e.g., 3090, 4090).
* PyTorch 2.0 is Required (for benchmarking only)

Verifying CUDA Versions:
* Run `nvcc --version` to confirm your CUDA toolkit version. The Docker container we provide includes version 12.1.
* Run `nvidia-smi` to verify your CUDA driver version. Our Docker container also uses version 12.1.

You can install from source:
```
cd src
nvcc ./1D/1d_main.cu -o 1d.out -lcufft -O3 --use_fast_math --gpu-architecture=sm_xx
nvcc ./2D/2d_main.cu -o 2d.out -lcufft -O3 --use_fast_math --gpu-architecture=sm_xx
nvcc ./3D/3d_main.cu -o 3d.out -lcufft -O3 --use_fast_math --gpu-architecture=sm_xx
```

  


