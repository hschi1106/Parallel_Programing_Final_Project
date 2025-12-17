// gpg_cpu.hpp
#pragma once

#include "gpg_types.hpp"
#include <random>

// ========== Utilities for programs ==========

std::string token_to_string(int tok);
std::string program_to_postfix_string(const std::vector<int> &prog);
std::string program_to_infix_string(const std::vector<int> &prog);
void print_FOS(const FOS &fos);

// ========== Program evaluation ==========

// Evaluate one program on one sample using a simple stack-based VM.
// If the program is invalid (stack underflow, wrong final stack size, NaN),
// we return a large penalty.
double eval_program_single(const std::vector<int> &prog, double x);

// Evaluate fitness (MSE) of one program on the whole dataset.
double evaluate_fitness_cpu(const std::vector<int> &prog, const Dataset &data);
double evaluate_fitness(const std::vector<int> &prog, const Dataset &data, bool is_gpu = false);

// Evaluate fitness for the whole population.
void evaluate_population_cpu(Population &pop, const Dataset &data);

// ========== GP functions ==========

// Generate a syntactically valid postfix program with only binary operators.
// genome_len must be odd.
std::vector<int> random_program(int genome_len, std::mt19937 &rng);

// Create a random *syntactically valid* postfix program with only binary operators.
// Requirement: genome_len should be odd (L = 2 * num_funcs + 1).
Individual random_individual(int genome_len, std::mt19937 &rng, const Dataset &data);

// Create a synthetic dataset using the target program with some noise.
Dataset make_synthetic_dataset(int n_samples, const std::vector<int> &target_prog,
                               const std::pair<double, double> x_range, const double noise_bound,
                               std::mt19937 &rng);

// Univariate FOS: each position alone
FOS make_univariate_fos(int genome_len);

// Compute pairwise mutual information between genome positions
// using discrete token values in the current population.
std::vector<std::vector<double>> compute_mutual_information_matrix(const Population &pop, int genome_len);

// Build a linkage-tree-style FOS from the current population,
// using mutual information as similarity between variables.
FOS build_linkage_tree_fos(const Population &pop, int genome_len);

// One GOMEA generation using given FOS.
void gomea_step(Population &pop, const FOS &fos, const Dataset &data, std::mt19937 &rng);
