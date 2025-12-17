// gp_cuda.hpp
#pragma once

#include "gpg_types.hpp"
#include <cuda_runtime.h>

void gpu_init(const Dataset &data);

void gpu_free();
