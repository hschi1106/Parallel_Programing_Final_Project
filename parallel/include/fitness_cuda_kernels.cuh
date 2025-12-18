#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA-only helpers. Include this ONLY from .cu files.

#define CUDA_CHECK(x)                                                  \
    do                                                                 \
    {                                                                  \
        cudaError_t err = (x);                                         \
        if (err != cudaSuccess)                                        \
        {                                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                              \
        }                                                              \
    } while (0)

__device__ bool finite_dev(double x);
__device__ double eval_program_single_dev(const int *prog, int prog_len,
                                          const double *inputs, int input_dim);
__global__ void fitness_kernel_single_prog_kernel(const int *d_prog, int prog_len,
                                                  const double *d_X, const double *d_y,
                                                  int N, int D, double *d_sum_out);
// Batched kernel: one kernel launch evaluates multiple programs.
__global__ void fitness_kernel_batch_kernel(const int *d_progs, int prog_len,
                                            const double *d_X, const double *d_y,
                                            int N, int D, int batch, double *d_sums_out);
