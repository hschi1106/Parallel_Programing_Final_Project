// fitness_cuda_kernels.cuh
#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <curand_kernel.h> // 需要 cuRAND

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

// Single program eval helper (Device only)
__device__ double eval_program_shared_mem(const int *s_genome, int prog_len,
                                          const double *inputs, int input_dim);

// RNG Init Kernel
__global__ void init_rng_kernel(curandState *states, unsigned long seed, int pop_size);

// The Main GOMEA Kernel
__global__ void gomea_generation_kernel(
    int *d_genomes,          // [POP_SIZE * LEN] (Global Memory)
    double *d_fitnesses,     // [POP_SIZE]
    const double *d_X,       // [N * D]
    const double *d_y,       // [N]
    const int *d_fos_data,   // Flattened FOS indices
    const int *d_fos_offsets,// FOS offsets
    const int *d_fos_sizes,  // FOS sizes
    int num_subsets,         // Number of subsets in FOS
    int pop_size,
    int prog_len,
    int N,
    int D,
    curandState *rng_states  // RNG states
);

// Single program eval wrapper (for compatibility)
__global__ void fitness_kernel_single_prog_kernel(const int *d_prog, int prog_len,
                                                  const double *d_X, const double *d_y,
                                                  int N, int D, double *d_sum_out);