// gpg_cuda.hpp
#pragma once
#include "gpg_types.hpp"
#include <vector>

void gpu_init(const Dataset &data, int pop_size, int genome_len, unsigned int seed);
void gpu_free();
void gpu_load_population(const Population &pop);
void gpu_retrieve_population(Population &pop);
void gpu_retrieve_fitnesses(std::vector<double> &fitnesses);
std::vector<double> gpu_compute_mi(int pop_size, int genome_len);
void gpu_run_gomea_generation(const FOS &fos, int pop_size, int genome_len, int n_samples);
double evaluate_fitness_gpu(const std::vector<int> &prog, const Dataset &data);

// 新增：印出累計的時間統計
void gpu_print_profile_stats();