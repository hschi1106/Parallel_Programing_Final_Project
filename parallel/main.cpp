// main.cpp
// ... 前面 includes 與原本相同 ...
#include "gpg_types.hpp"
#include "gpg_cpu.hpp"
#include "gpg_cuda.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>
int main()
{
    // ... 前面設定不變 ...
    const int POP_SIZE = 4096;
    const int GENOME_LEN = 31;
    const int TARGET_LEN = 31;
    const int N_SAMPLES = 128;
    const int MAX_GENERATIONS = 10;
    const unsigned SEED = 123456u;
    auto start_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(SEED);
    
    std::vector<int> target_prog = random_program(TARGET_LEN, rng);
    std::cout << "Target program (postfix): " << program_to_postfix_string(target_prog) << "\n";

    Dataset data = make_synthetic_dataset(N_SAMPLES, target_prog, std::make_pair(-100.0, 100.0), 0.1, rng);

    Population pop;
    pop.reserve(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i) pop.push_back(random_individual(GENOME_LEN, rng, data));
    
    std::cout << "Initializing GPU...\n";
    gpu_init(data, POP_SIZE, GENOME_LEN, SEED);
    gpu_load_population(pop); 
    
    std::vector<double> current_fitnesses(POP_SIZE);
    gpu_retrieve_fitnesses(current_fitnesses);
    double best_fit = *std::min_element(current_fitnesses.begin(), current_fitnesses.end());
    std::cout << "Initial best fitness: " << best_fit << '\n';

    for (int gen = 0; gen < MAX_GENERATIONS; ++gen)
    {
        std::vector<double> mi_matrix = gpu_compute_mi(POP_SIZE, GENOME_LEN);
        
        gpu_retrieve_population(pop); 
        FOS fos = build_linkage_tree_fos(pop, GENOME_LEN); 

        gpu_run_gomea_generation(fos, POP_SIZE, GENOME_LEN, N_SAMPLES);

        gpu_retrieve_fitnesses(current_fitnesses);
        best_fit = *std::min_element(current_fitnesses.begin(), current_fitnesses.end());
        
        std::cout << "Gen " << gen + 1 << ": best fitness = " << best_fit << '\n';
        if (best_fit < 1e-5) break;
    }

    gpu_retrieve_population(pop);
    
    // 印出 GPU 效能統計
    gpu_print_profile_stats();
    
    gpu_free();

    auto best_it = std::min_element(pop.begin(), pop.end(),
                            [](const Individual &a, const Individual &b)
                            { return a.fitness < b.fitness; });
                            
    std::cout << "Done. Final best fitness: " << best_it->fitness << '\n';
    std::cout << "Best program (postfix): " << program_to_postfix_string(best_it->genome) << "\n";
    std::cout << "Best program (infix):   "
              << program_to_infix_string(best_it->genome) << "\n";

    std::cout << "Target program (infix): "
              << program_to_infix_string(target_prog) << "\n";
    auto end_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    double total_seconds = (end_time - start_time) / 1e9;
    std::cout << "Total time: " << total_seconds << " seconds\n";

    return 0;
}