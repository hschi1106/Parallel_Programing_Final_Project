// main.cpp
#include "gpg_types.hpp"
#include "gpg_cpu.hpp"
#include "gpg_cuda.hpp"
#include <random>
#include <iostream>
#include <algorithm>

int main()
{
    const int POP_SIZE = 4096;
    const int GENOME_LEN = 31;
    const int TARGET_LEN = 31;
    const int N_SAMPLES = 128;
    const int MAX_GENERATIONS = 1000;
    const unsigned SEED = 123456u;

    std::mt19937 rng(SEED);

    // 1. Target program
    std::vector<int> target_prog = random_program(TARGET_LEN, rng);
    std::cout << "Target program (postfix): "
              << program_to_postfix_string(target_prog) << "\n";
    std::cout << "Target program (infix):   "
              << program_to_infix_string(target_prog) << "\n";

    // 2. Dataset
    Dataset data = make_synthetic_dataset(N_SAMPLES, target_prog, std::make_pair(-100.0, 100.0), 0.1, rng);

    // 3. Population init (CPU)
    Population pop;
    pop.reserve(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i)
    {
        pop.push_back(random_individual(GENOME_LEN, rng, data));
    }

    const int genome_len = (int)pop.front().genome.size();

    auto get_best = [&]()
    {
        return std::min_element(pop.begin(), pop.end(),
                                [](const Individual &a, const Individual &b)
                                {
                                    return a.fitness < b.fitness;
                                });
    };

    auto best_it = get_best();
    std::cout << "Initial best fitness: " << best_it->fitness << '\n';

    gpu_init(data);

    for (int gen = 0; gen < MAX_GENERATIONS; ++gen)
    {
        FOS fos = build_linkage_tree_fos(pop, genome_len);

        gomea_step(pop, fos, data, rng);

        best_it = get_best();
        std::cout << "Gen " << gen + 1 << ": best fitness = " << best_it->fitness << '\n';
    }

    gpu_free();

    best_it = get_best();
    std::cout << "Done. Final best fitness: " << best_it->fitness << '\n';

    std::cout << "Best program (postfix): "
              << program_to_postfix_string(best_it->genome) << "\n";
    std::cout << "Best program (infix):   "
              << program_to_infix_string(best_it->genome) << "\n";

    return 0;
}
