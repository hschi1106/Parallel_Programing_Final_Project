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

struct GpuEvalContext
{
    const Dataset *host_data = nullptr; // 用來做「這個 ctx 對應哪個 Dataset」的檢查
    int N = 0;
    int D = 0;
    int prog_len = 0;

    double *d_X = nullptr;   // [N*D]
    double *d_y = nullptr;   // [N]
    int *d_prog = nullptr;   // [prog_len]
    double *d_sum = nullptr; // [1]
};
