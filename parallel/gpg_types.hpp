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
    double x;
    double y;
};

using Dataset = std::vector<Sample>;

// Token encoding
enum Token : int
{
    VAR_X = 0,
    CONST_1 = 1,
    CONST_2 = 2,
    OP_ADD = 3,
    OP_SUB = 4,
    OP_MUL = 5,
    OP_DIV = 6,
    TOKEN_MIN = 0,
    TOKEN_MAX = 6
};

struct Individual
{
    std::vector<int> genome; // postfix tokens
    double fitness = std::numeric_limits<double>::infinity();
};

using Population = std::vector<Individual>;
using FOS = std::vector<std::vector<int>>;
