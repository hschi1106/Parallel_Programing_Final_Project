// main.cpp
#include "types.hpp"
#include "gomea.hpp"
#include "fitness_cuda.hpp"
#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>

int main(int argc, char **argv)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Hyperparameters
    const int POP_SIZE = 4096;
    const int GENOME_LEN = 15; // must be odd
    const int MAX_GENERATIONS = 10;

    // Fixed seed for reproducibility
    const unsigned SEED = 123456u;
    std::mt19937 rng(SEED);

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " train.txt test.txt operand_count\n";
        return 1;
    }

    size_t operand_count = std::stoul(argv[3]);

    // Load training data
    std::ifstream fin_train(argv[1]);
    if (!fin_train)
        return 1;

    Dataset train_data;
    std::string line;
    size_t line_count = 0;
    while (std::getline(fin_train, line))
    {
        ++line_count;
        std::istringstream iss(line);
        std::vector<double> values;
        double val;
        while (iss >> val)
        {
            values.push_back(val);
        }
        if (values.size() != operand_count + 1)
        {
            std::cerr << "Error: line " << line_count
                      << " has incorrect number of values (expected "
                      << (operand_count + 1) << ", got " << values.size() << ")\n";
            return 1;
        }
        Sample sample;
        sample.inputs.resize(operand_count);
        sample.inputs = std::vector<double>(values.begin(), values.begin() + operand_count);
        sample.output = values[operand_count];
        train_data.push_back(std::move(sample));
    }

    // Load testing data
    std::ifstream fin_test(argv[2]);
    if (!fin_test)
        return 1;
    Dataset test_data;
    line_count = 0;
    while (std::getline(fin_test, line))
    {
        ++line_count;
        std::istringstream iss(line);
        std::vector<double> values;
        double val;
        while (iss >> val)
        {
            values.push_back(val);
        }
        if (values.size() != operand_count + 1)
        {
            std::cerr << "Error: line " << line_count
                      << " has incorrect number of values (expected "
                      << (operand_count + 1) << ", got " << values.size() << ")\n";
            return 1;
        }
        Sample sample;
        sample.inputs.resize(operand_count);
        sample.inputs = std::vector<double>(values.begin(), values.begin() + operand_count);
        sample.output = values[operand_count];
        test_data.push_back(std::move(sample));
    }

    GpuEvalContext train_ctx;
    // gpu_eval_init(train_ctx, train_data, operand_count, GENOME_LEN);
    gpu_eval_init(train_ctx, train_data, operand_count, GENOME_LEN, POP_SIZE, SEED);

    Population pop;
    pop.reserve(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i)
    {
        pop.push_back(random_individual(GENOME_LEN, rng, train_data));
    }

    const int actual_genome_len = (int)pop.front().genome.size();

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

    for (int gen = 0; gen < MAX_GENERATIONS; ++gen)
    {
        FOS fos = build_linkage_tree_fos(pop, actual_genome_len);

        gomea_step(pop, fos, train_data, rng, &train_ctx);
        best_it = get_best();
        std::cout << "Gen " << gen + 1 << ": best fitness = " << best_it->fitness << '\n';
    }

    // Evaluate best on test set
    double test_fitness = evaluate_fitness(best_it->genome, test_data, nullptr);
    std::cout << "Best test fitness: " << test_fitness << '\n';

    // Print final program
    std::cout << "Best program (postfix): "
              << program_to_postfix_string(best_it->genome) << "\n";

    gpu_eval_destroy(train_ctx);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total elapsed time: " << elapsed.count() << " seconds\n";

    return 0;
}
