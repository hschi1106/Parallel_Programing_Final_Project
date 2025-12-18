// types.hpp
#pragma once

#include <vector>
#include <string>
#include <limits>
#include <sstream>
#include <cmath>
#include <array>

struct Sample
{
    std::vector<double> inputs;
    double output;
};

using Dataset = std::vector<Sample>;

enum Token : int
{
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2,
    OP_DIV = 3,
    OP_SIN = 4,
    OP_COS = 5,
    OP_EXP = 6,
    VAR_1 = 7,
    VAR_2 = 8,
    VAR_3 = 9,
    TOKEN_MIN = 0,
    TOKEN_MAX = 9
};

struct Individual
{
    std::vector<int> genome; // token sequence (postfix)
    double fitness = std::numeric_limits<double>::infinity();
};

using Population = std::vector<Individual>;
using FOS = std::vector<std::vector<int>>; // Family of Subsets

// Forward declaration for curandState
struct TRngState; 

struct GpuEvalContext
{
    const Dataset *host_data = nullptr; 
    int N = 0;
    int D = 0;
    int prog_len = 0;
    int pop_size = 0; // 新增：紀錄族群大小

    // Dataset pointers
    double *d_X = nullptr;   // [N*D]
    double *d_y = nullptr;   // [N]
    
    // Single Eval buffers (保留給 random_individual 初始評估用)
    int *d_prog_single = nullptr;   
    double *d_sum_single = nullptr; 

    // Population Evolution Buffers (新增)
    int *d_pop_genomes = nullptr;    // [POP_SIZE * PROG_LEN] flattened
    double *d_pop_fitness = nullptr; // [POP_SIZE]
    void *d_rng_states = nullptr;    // curandState* (cast to void* here to avoid cuda header dependency)

    // FOS Buffers (新增 - 每次 gomea_step 重複利用)
    int *d_fos_data = nullptr;       // Flattened indices of all subsets
    int *d_fos_offsets = nullptr;    // Start index of each subset
    int *d_fos_sizes = nullptr;      // Size of each subset
    int max_fos_nodes = 0;           // Capacity of fos buffers
};