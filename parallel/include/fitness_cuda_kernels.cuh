#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA-only helpers. Include this ONLY from .cu files.

#define CUDA_CHECK(x)                                                     \
    do                                                                    \
    {                                                                     \
        cudaError_t err = (x);                                            \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)        \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)
