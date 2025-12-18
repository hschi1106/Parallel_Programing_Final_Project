#include "gpg_types.hpp"
#include "gpg_cpu.hpp"
#include "gpg_cuda.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>

std::string token_to_string(int tok) {
    switch (tok) {
    case OP_ADD: return "+";
    case OP_SUB: return "-";
    case OP_MUL: return "*";
    case OP_DIV: return "/";
    case OP_SIN: return "sin";
    case OP_COS: return "cos";
    case OP_EXP: return "exp";
    default:
        if (tok >= VAR_1) {
            // 將 VAR_1 轉為 "x0", VAR_2 轉為 "x1"... 以此類推
            return "x" + std::to_string(tok - VAR_1);
        }
        return "?";
    }
}

std::string program_to_postfix_string(const std::vector<int> &prog) {
    std::ostringstream oss;
    for (size_t i = 0; i < prog.size(); ++i) {
        if (i > 0) oss << ' ';
        oss << token_to_string(prog[i]);
    }
    return oss.str();
}

std::string program_to_infix_string(const std::vector<int> &prog) {
    std::vector<std::string> st;
    st.reserve(32);

    for (int tok : prog) {
        if (tok >= VAR_1) {
            // 變數 (Operand) -> 直接推入堆疊
            st.push_back(token_to_string(tok));
        } 
        else if (tok == OP_SIN || tok == OP_COS || tok == OP_EXP) {
            // 一元運算子 (Unary) -> 需要 1 個運算元
            if (st.empty()) return "<invalid postfix program>";
            
            std::string a = std::move(st.back());
            st.pop_back();

            // 格式：op(a)
            std::string op = token_to_string(tok);
            st.push_back(op + "(" + a + ")");
        } 
        else if (tok == OP_ADD || tok == OP_SUB || tok == OP_MUL || tok == OP_DIV) {
            // 二元運算子 (Binary) -> 需要 2 個運算元
            if (st.size() < 2) return "<invalid postfix program>";

            std::string rhs = std::move(st.back());
            st.pop_back();
            std::string lhs = std::move(st.back());
            st.pop_back();

            // 格式：(lhs op rhs)
            std::string op = token_to_string(tok);
            st.push_back("(" + lhs + " " + op + " " + rhs + ")");
        } 
        else {
            return "<unknown token>";
        }
    }

    if (st.size() != 1) {
        return "<invalid postfix program>";
    }
    return st.back();
}

int main(int argc, char **argv)
{
    
    // Hyperparameters (Match Serial Code)
    const int POP_SIZE = 4096;
    const int GENOME_LEN = 31; 
    const int MAX_GENERATIONS = 20;
    const unsigned SEED = 123456u;
    std::mt19937 rng(SEED);
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " train.txt test.txt operand_count\n";
        return 1;
    }

    size_t operand_count = std::stoul(argv[3]);

    // 1. Load Data (Using CPU logic implemented in gpg_cpu.cpp)
    Dataset train_data = load_data(argv[1], operand_count);
    Dataset test_data = load_data(argv[2], operand_count);

 

    // 2. Initialize Population (CPU)
    Population pop;
    pop.reserve(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i) {
        pop.push_back(random_individual(GENOME_LEN, rng, train_data));
    }

    auto get_best = [&]() {
        return std::min_element(pop.begin(), pop.end(),
            [](const Individual &a, const Individual &b) { return a.fitness < b.fitness; });
    };

    auto best_it = get_best();
    std::cout << "Initial best fitness: " << best_it->fitness << '\n';

    // 3. Initialize GPU
    gpu_init(train_data, POP_SIZE, GENOME_LEN, SEED);
    gpu_load_population(pop);

    std::vector<double> fitnesses(POP_SIZE);

    // 4. Evolution Loop
    for (int gen = 0; gen < MAX_GENERATIONS; ++gen)
    {
        // A. MI (GPU)
        std::vector<double> mi = gpu_compute_mi(POP_SIZE, GENOME_LEN);

        // B. Linkage Tree (CPU)
        FOS fos = build_linkage_tree_from_mi(mi, GENOME_LEN);

        // C. GOMEA Step (GPU)
        gpu_run_gomea_generation_block_parallel(fos, POP_SIZE, GENOME_LEN);

        // D. Sync Stats
        gpu_retrieve_fitnesses(fitnesses);
        double best_fit = *std::min_element(fitnesses.begin(), fitnesses.end());
        std::cout << "Gen " << gen + 1 << ": best fitness = " << best_fit << '\n';
    }

    // 5. Finalize
    gpu_retrieve_population(pop);
    gpu_print_profile_stats();
    gpu_free();

    // Evaluate best on test set (CPU)
    best_it = get_best();
    double test_fitness = evaluate_fitness_cpu(best_it->genome, test_data);
    std::cout << "Best test fitness: " << test_fitness << '\n';
    // Print final program
    std::cout << "Best program (postfix): "
              << program_to_postfix_string(best_it->genome) << "\n";
    // Print input program
    std::cout << "Best program (infix):   "
              << program_to_infix_string(best_it->genome) << "\n";


    return 0;
}