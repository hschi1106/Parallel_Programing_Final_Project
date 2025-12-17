// gpg_types.hpp
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
