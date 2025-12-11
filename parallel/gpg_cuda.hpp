// gp_cuda.hpp
#pragma once

#include "gpg_types.hpp"

void gpu_init(const Dataset &data);

void gpu_free();

double evaluate_fitness_gpu(const std::vector<int> &prog, const Dataset &data);
