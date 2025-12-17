#pragma once

#include "types.hpp"
#include <vector>

// Host-facing CUDA fitness evaluator API.
// NOTE: This header must stay pure C++ (no cuda_runtime.h, no __device__/__global__).

void gpu_eval_init(GpuEvalContext &ctx, const Dataset &data, int operand_count, int prog_len);
void gpu_eval_destroy(GpuEvalContext &ctx);

double evaluate_fitness_gpu(GpuEvalContext &ctx, const std::vector<int> &prog);
