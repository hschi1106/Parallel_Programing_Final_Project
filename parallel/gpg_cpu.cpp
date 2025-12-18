#include "gpg_types.hpp"
#include "gpg_cpu.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

// ========== 檔案讀取 (完全複製原始 main 的邏輯) ==========
Dataset load_data(const std::string& filename, int operand_count) {
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Error: Cannot open " << filename << "\n";
        exit(1);
    }
    Dataset data;
    std::string line;
    size_t line_count = 0;
    while (std::getline(fin, line)) {
        ++line_count;
        std::istringstream iss(line);
        std::vector<double> values;
        double val;
        while (iss >> val) {
            values.push_back(val);
        }
        if (values.size() != (size_t)(operand_count + 1)) {
            std::cerr << "Error: line " << line_count << " has incorrect values.\n";
            exit(1);
        }
        Sample sample;
        sample.inputs.assign(values.begin(), values.begin() + operand_count);
        sample.output = values[operand_count];
        data.push_back(std::move(sample));
    }
    return data;
}

// ========== CPU 評估 (Reference) ==========
// 邏輯與 Serial Code 完全一致：支援 Unary 但保護 Division/Exp
double eval_program_single_cpu(const std::vector<int> &prog, const std::vector<double>& inputs) {
    static const double PENALTY = 1e6;
    std::vector<double> stack;
    stack.reserve(32);

    for (int tok : prog) {
        if (tok >= VAR_1) {
            int var_idx = tok - VAR_1;
            if (var_idx < 0 || var_idx >= (int)inputs.size()) return PENALTY;
            stack.push_back(inputs[var_idx]);
        } else {
            switch (tok) {
                case OP_ADD:
                case OP_SUB:
                case OP_MUL:
                case OP_DIV:
                {
                    if (stack.size() < 2) return PENALTY;
                    double b = stack.back(); stack.pop_back();
                    double a = stack.back(); stack.pop_back();
                    double r = 0.0;
                    if (tok == OP_ADD) r = a + b;
                    else if (tok == OP_SUB) r = a - b;
                    else if (tok == OP_MUL) r = a * b;
                    else { // OP_DIV
                        if (std::fabs(b) >= 0.001) r = a / b;
                        else r = 1.0; 
                    }
                    if (!std::isfinite(r)) return PENALTY;
                    stack.push_back(r);
                    break;
                }
                case OP_SIN:
                case OP_COS:
                case OP_EXP:
                {
                    if (stack.empty()) return PENALTY;
                    double a = stack.back(); stack.pop_back();
                    double r = 0.0;
                    if (tok == OP_SIN) r = std::sin(a);
                    else if (tok == OP_COS) r = std::cos(a);
                    else { // OP_EXP
                        if (a <= 10.0) r = std::exp(a);
                        else r = std::exp(10.0);
                    }
                    if (!std::isfinite(r)) return PENALTY;
                    stack.push_back(r);
                    break;
                }
                default:
                    return PENALTY;
            }
        }
    }
    if (stack.size() != 1) return PENALTY;
    double v = stack.back();
    return std::isfinite(v) ? v : PENALTY;
}

double evaluate_fitness_cpu(const std::vector<int> &prog, const Dataset &data) {
    double sum = 0.0;
    for (const auto &s : data) {
        double y_hat = eval_program_single_cpu(prog, s.inputs);
        double diff = y_hat - s.output;
        sum += diff * diff;
    }
    return sum / static_cast<double>(data.size()); // MSE
}

// ========== 初始化 ==========
// 嚴格遵照 Serial Code：只生成二元運算子 (Binary Operators Only)
std::vector<int> random_program(int genome_len, std::mt19937 &rng, int num_inputs) {
    if (genome_len % 2 == 0) genome_len -= 1;
    const int num_funcs = (genome_len - 1) / 2;
    const int num_operands = num_funcs + 1;
    int used_funcs = 0, used_operands = 0, stack_depth = 0;

    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::vector<int> prog;
    prog.reserve(genome_len);

    for (int pos = 0; pos < genome_len; ++pos) {
        bool choose_operand = false;
        if (used_operands == num_operands) choose_operand = false;
        else if (used_funcs == num_funcs) choose_operand = true;
        else if (stack_depth < 2) choose_operand = true;
        else choose_operand = (coin(rng) < 0.5);

        if (choose_operand) {
            std::uniform_int_distribution<int> op_dist(VAR_1, VAR_1 + num_inputs - 1);
            prog.push_back(op_dist(rng));
            used_operands++; stack_depth++;
        } else {
            // Serial Code Line 192: 僅生成 OP_ADD 到 OP_DIV
            std::uniform_int_distribution<int> f_dist(OP_ADD, OP_DIV);
            prog.push_back(f_dist(rng));
            used_funcs++; stack_depth--; // Binary consumes 2, pushes 1
        }
    }
    return prog;
}

Individual random_individual(int genome_len, std::mt19937 &rng, const Dataset &data) {
    Individual ind;
    ind.genome = random_program(genome_len, rng, (int)data[0].inputs.size());
    ind.fitness = evaluate_fitness_cpu(ind.genome, data);
    return ind;
}

// ========== FOS 建構 (接收 MI) ==========
FOS build_linkage_tree_from_mi(const std::vector<double>& flat_mi, int genome_len) {
    FOS fos;
    std::vector<std::vector<int>> clusters;
    for (int i = 0; i < genome_len; ++i) clusters.push_back({i});
    fos = clusters;

    auto get_mi = [&](int i, int j) { return flat_mi[i * genome_len + j]; };

    while (clusters.size() > 1) {
        double best_score = -1e9;
        int best_a = -1, best_b = -1;

        for (int a = 0; a < (int)clusters.size(); ++a) {
            for (int b = a + 1; b < (int)clusters.size(); ++b) {
                double sum = 0.0;
                int cnt = 0;
                for (int i : clusters[a]) {
                    for (int j : clusters[b]) {
                        sum += get_mi(i, j);
                        cnt++;
                    }
                }
                double avg = (cnt > 0) ? (sum / cnt) : 0.0;
                if (avg > best_score) {
                    best_score = avg; best_a = a; best_b = b;
                }
            }
        }
        if (best_a == -1) break;

        std::vector<int> merged = clusters[best_a];
        merged.insert(merged.end(), clusters[best_b].begin(), clusters[best_b].end());
        std::sort(merged.begin(), merged.end());
        fos.push_back(merged);

        clusters[best_a] = merged;
        clusters.erase(clusters.begin() + best_b);
    }
    return fos;
}