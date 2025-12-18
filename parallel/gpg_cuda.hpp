#pragma once
#include "gpg_types.hpp"
#include <vector>

// 初始化：分配記憶體、轉移資料
// 注意：為了支援多變數，這裡不再只傳 vector<Sample>，而是讓實作內部轉為 Flat Array
void gpu_init(const Dataset &data, int pop_size, int genome_len, unsigned int seed);
void gpu_free();

// 資料傳輸
void gpu_load_population(const Population &pop);
void gpu_retrieve_population(Population &pop);
void gpu_retrieve_fitnesses(std::vector<double> &fitnesses);

// 計算互信息
std::vector<double> gpu_compute_mi(int pop_size, int genome_len);

// GOMEA 演化
void gpu_run_gomea_generation_block_parallel(const FOS &fos, int pop_size, int genome_len);

// Profiling
void gpu_print_profile_stats();