#pragma once

#include <vector>
#include <string>
#include <limits>
#include <cmath>

// 保持與原始 Serial Code 一致的 Token 定義
enum Token : int
{
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2,
    OP_DIV = 3,
    OP_SIN = 4,
    OP_COS = 5,
    OP_EXP = 6,
    VAR_1  = 7,
    // VAR_2, VAR_3... 會根據 operand_count 動態延伸
    
    TOKEN_MIN = 0,
    // TOKEN_MAX 根據實際使用情況判斷，在此用於 MI 計算的 Alphabet Size
    // 假設最多支援 10 個變數，TOKEN_MAX 約為 17
    MAX_TOKENS_SUPPORTED = 20 
};

struct Sample
{
    std::vector<double> inputs; // 支援多維輸入
    double output;
};

using Dataset = std::vector<Sample>;

struct Individual
{
    std::vector<int> genome; // postfix tokens
    double fitness = std::numeric_limits<double>::infinity();
};

using Population = std::vector<Individual>;
using FOS = std::vector<std::vector<int>>;