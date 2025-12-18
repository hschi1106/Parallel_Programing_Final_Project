#pragma once
#include "gpg_types.hpp"
#include <random>
#include <string>
#include <vector>

// 檔案 I/O 與字串處理 (對應原始 main 的讀取邏輯)
Dataset load_data(const std::string& filename, int operand_count);

// 程式碼轉換與評估 (CPU Reference)
double eval_program_single_cpu(const std::vector<int> &prog, const std::vector<double>& inputs);
double evaluate_fitness_cpu(const std::vector<int> &prog, const Dataset &data);

// 初始化與 Linkage Learning
std::vector<int> random_program(int genome_len, std::mt19937 &rng, int num_inputs);
Individual random_individual(int genome_len, std::mt19937 &rng, const Dataset &data);

// FOS 建構 (CPU 負責樹的建構，MI 矩陣由 GPU 計算)
FOS build_linkage_tree_from_mi(const std::vector<double>& flat_mi, int genome_len);