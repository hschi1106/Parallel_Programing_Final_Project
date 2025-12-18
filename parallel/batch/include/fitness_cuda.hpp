#pragma once

#include "types.hpp"
#include <vector>

// Host-facing CUDA fitness evaluator API.
// NOTE: This header must stay pure C++ (no cuda_runtime.h, no __device__/__global__).

void gpu_eval_init(GpuEvalContext &ctx, const Dataset &data, int operand_count, int prog_len);
void gpu_eval_destroy(GpuEvalContext &ctx);

double evaluate_fitness_gpu(GpuEvalContext &ctx, const std::vector<int> &prog);

// Batched evaluation: evaluate `batch` programs (flattened) in one GPU call.
// h_progs_flat layout: [batch][prog_len] contiguous.
void evaluate_fitness_gpu_batch(GpuEvalContext &ctx, const int *h_progs_flat, int batch, double *h_out);
