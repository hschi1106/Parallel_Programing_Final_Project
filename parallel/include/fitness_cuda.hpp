#pragma once

#include "types.hpp"
#include <vector>

void gpu_eval_init(GpuEvalContext &ctx, const Dataset &data, int operand_count, int prog_len, int pop_size, unsigned int seed);
void gpu_eval_destroy(GpuEvalContext &ctx);

// 單一程式評估 (保留給初始化使用)
double evaluate_fitness_gpu_single(GpuEvalContext &ctx, const std::vector<int> &prog);

// 核心：全族群 GOMEA 演化
void gpu_run_gomea_generation(GpuEvalContext &ctx, const FOS &fos);

// 資料傳輸 helper
void gpu_load_population(GpuEvalContext &ctx, const Population &pop);
void gpu_retrieve_population(GpuEvalContext &ctx, Population &pop);