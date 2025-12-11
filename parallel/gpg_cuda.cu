// gpg_cuda.cu
#include "gpg_types.hpp"
#include "gpg_cuda.hpp"
#include <iostream>

void gpu_init(const Dataset &data)
{
    (void)data;
}

void gpu_free()
{
}

double evaluate_fitness_gpu(const std::vector<int> &prog, const Dataset &data)
{
    return 0.0;
}
