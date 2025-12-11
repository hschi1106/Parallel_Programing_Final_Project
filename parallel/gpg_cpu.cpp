// gpg_cpu.cpp
#include "gpg_types.hpp"
#include "gpg_cpu.hpp"
#include "gpg_cuda.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>

// ========== Utilities for programs ==========
std::string token_to_string(int tok)
{
    switch (tok)
    {
    case VAR_X:
        return "x";
    case CONST_1:
        return "1";
    case CONST_2:
        return "2";
    case OP_ADD:
        return "+";
    case OP_SUB:
        return "-";
    case OP_MUL:
        return "*";
    case OP_DIV:
        return "/";
    default:
        return "?";
    }
}

std::string program_to_postfix_string(const std::vector<int> &prog)
{
    std::ostringstream oss;
    for (size_t i = 0; i < prog.size(); ++i)
    {
        if (i > 0)
            oss << ' ';
        oss << token_to_string(prog[i]);
    }
    return oss.str();
}

std::string program_to_infix_string(const std::vector<int> &prog)
{
    std::vector<std::string> st;
    st.reserve(32);

    auto is_operator = [](int tok)
    {
        return tok == OP_ADD || tok == OP_SUB || tok == OP_MUL || tok == OP_DIV;
    };

    auto op_to_string = [](int tok) -> std::string
    {
        switch (tok)
        {
        case OP_ADD:
            return "+";
        case OP_SUB:
            return "-";
        case OP_MUL:
            return "*";
        case OP_DIV:
            return "/";
        default:
            return "?";
        }
    };

    for (int tok : prog)
    {
        if (!is_operator(tok))
        {
            st.push_back(token_to_string(tok));
        }
        else
        {
            if (st.size() < 2)
                return "<invalid postfix program>";

            std::string rhs = std::move(st.back());
            st.pop_back();
            std::string lhs = std::move(st.back());
            st.pop_back();

            std::string expr = "(" + lhs + " " + op_to_string(tok) + " " + rhs + ")";
            st.push_back(std::move(expr));
        }
    }

    if (st.size() != 1)
        return "<invalid postfix program>";
    return st.back();
}

[[maybe_unused]]
void print_FOS(const FOS &fos)
{
    std::cout << "FOS subsets:\n";
    for (size_t i = 0; i < fos.size(); ++i)
    {
        std::cout << "  Subset " << i << ": { ";
        for (int pos : fos[i])
        {
            std::cout << pos << " ";
        }
        std::cout << "}\n";
    }
}

// ========== Program evaluation ==========
double eval_program_single(const std::vector<int> &prog, double x)
{
    static double PENALTY = 1e6;
    std::vector<double> stack;
    stack.reserve(32);

    for (int tok : prog)
    {
        switch (tok)
        {
        case VAR_X:
            stack.push_back(x);
            break;
        case CONST_1:
            stack.push_back(1.0);
            break;
        case CONST_2:
            stack.push_back(2.0);
            break;
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        {
            if (stack.size() < 2)
                return PENALTY;
            double b = stack.back();
            stack.pop_back();
            double a = stack.back();
            stack.pop_back();
            double r = 0.0;
            if (tok == OP_ADD)
                r = a + b;
            else if (tok == OP_SUB)
                r = a - b;
            else if (tok == OP_MUL)
                r = a * b;
            else
            {
                // protected division
                if (std::fabs(b) < 1e-9)
                    r = a;
                else
                    r = a / b;
            }
            if (!std::isfinite(r))
                return PENALTY;
            stack.push_back(r);
            break;
        }
        default:
            // Unknown token -> penalty
            return PENALTY;
        }
    }

    if (stack.size() != 1)
        return PENALTY;
    double v = stack.back();
    if (!std::isfinite(v))
        return PENALTY;
    return v;
}

double evaluate_fitness_cpu(const std::vector<int> &prog, const Dataset &data)
{
    double sum = 0.0;
    for (const auto &s : data)
    {
        double y_hat = eval_program_single(prog, s.x);
        double diff = y_hat - s.y;
        sum += diff * diff;
        if (!std::isfinite(sum))
            return 1e12;
    }
    return sum;
}

double evaluate_fitness(const std::vector<int> &prog, const Dataset &data, bool is_gpu = false)
{
    if (is_gpu)
    {
        return evaluate_fitness_gpu(prog, data);
    }
    else
    {
        return evaluate_fitness_cpu(prog, data);
    }
}

[[maybe_unused]]
void evaluate_population_cpu(Population &pop, const Dataset &data)
{
    for (auto &ind : pop)
    {
        ind.fitness = evaluate_fitness_cpu(ind.genome, data);
    }
}

// ========== GP functions ==========
std::vector<int> random_program(int genome_len, std::mt19937 &rng)
{
    if (genome_len % 2 == 0)
    {
        genome_len -= 1; // enforce odd length
    }

    const int num_funcs = (genome_len - 1) / 2;
    const int num_operands = num_funcs + 1;

    int used_funcs = 0;
    int used_operands = 0;
    int stack_depth = 0;

    std::uniform_real_distribution<double> coin(0.0, 1.0);

    std::vector<int> prog;
    prog.reserve(genome_len);

    for (int pos = 0; pos < genome_len; ++pos)
    {
        bool choose_operand = false;

        if (used_operands == num_operands)
        {
            choose_operand = false; // no operands left
        }
        else if (used_funcs == num_funcs)
        {
            choose_operand = true; // no functions left
        }
        else if (stack_depth < 2)
        {
            choose_operand = true; // need more stack before using binary op
        }
        else
        {
            choose_operand = (coin(rng) < 0.5);
        }

        if (choose_operand)
        {
            std::uniform_int_distribution<int> op_dist(0, 2); // x, 1, 2
            int which = op_dist(rng);
            int tok = VAR_X;
            if (which == 1)
                tok = CONST_1;
            else if (which == 2)
                tok = CONST_2;
            prog.push_back(tok);
            used_operands++;
            stack_depth++;
        }
        else
        {
            std::uniform_int_distribution<int> f_dist(OP_ADD, OP_DIV);
            int tok = f_dist(rng);
            prog.push_back(tok);
            used_funcs++;
            stack_depth--; // consume 2, push 1
        }
    }

    return prog;
}

Individual random_individual(int genome_len, std::mt19937 &rng, const Dataset &data)
{
    Individual ind;
    ind.genome = random_program(genome_len, rng);
    ind.fitness = evaluate_fitness(ind.genome, data);
    return ind;
}

Dataset make_synthetic_dataset(int n_samples, const std::vector<int> &target_prog,
                               const std::pair<double, double> x_range, const double noise_bound,
                               std::mt19937 &rng)
{
    Dataset data;
    data.reserve(n_samples);
    std::uniform_real_distribution<double> dist_x(x_range.first, x_range.second);
    std::normal_distribution<double> noise(0.0, noise_bound);

    for (int i = 0; i < n_samples; ++i)
    {
        double x = dist_x(rng);
        double y_clean = eval_program_single(target_prog, x);
        double y = y_clean + noise(rng);
        data.push_back({x, y});
    }
    return data;
}

[[maybe_unused]]
FOS make_univariate_fos(int genome_len)
{
    FOS fos;
    fos.reserve(genome_len);
    for (int i = 0; i < genome_len; ++i)
    {
        fos.push_back({i});
    }
    return fos;
}

std::vector<std::vector<double>> compute_mutual_information_matrix(const Population &pop, int genome_len)
{
    const int n = genome_len;
    std::vector<std::vector<double>> mi(n, std::vector<double>(n, 0.0));

    if (pop.empty())
        return mi;

    const int ALPHABET_SIZE = TOKEN_MAX - TOKEN_MIN + 1;
    const double N = static_cast<double>(pop.size());

    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            std::array<int, ALPHABET_SIZE> count_i{};
            std::array<int, ALPHABET_SIZE> count_j{};
            std::array<std::array<int, ALPHABET_SIZE>, ALPHABET_SIZE> count_ij{};

            // count frequencies over population
            for (const auto &ind : pop)
            {
                int vi = ind.genome[i] - TOKEN_MIN;
                int vj = ind.genome[j] - TOKEN_MIN;
                if (vi < 0 || vi >= ALPHABET_SIZE ||
                    vj < 0 || vj >= ALPHABET_SIZE)
                    continue; // ignore out-of-range tokens

                count_i[vi]++;
                count_j[vj]++;
                count_ij[vi][vj]++;
            }

            double mi_ij = 0.0;

            for (int a = 0; a < ALPHABET_SIZE; ++a)
            {
                double pi = count_i[a] / N;
                if (pi <= 0.0)
                    continue;

                for (int b = 0; b < ALPHABET_SIZE; ++b)
                {
                    double pj = count_j[b] / N;
                    int cij = count_ij[a][b];
                    if (pj <= 0.0 || cij == 0)
                        continue;

                    double pij = cij / N;
                    double denom = pi * pj;
                    if (denom <= 0.0)
                        continue;

                    double ratio = pij / denom;
                    mi_ij += pij * std::log(ratio); // nat log; base doesn't matter
                }
            }

            if (mi_ij < 0.0)
                mi_ij = 0.0; // numerical noise clamp

            mi[i][j] = mi_ij;
            mi[j][i] = mi_ij;
        }
    }

    return mi;
}

FOS build_linkage_tree_fos(const Population &pop, int genome_len)
{
    FOS fos;
    if (genome_len <= 0 || pop.empty())
        return fos;

    // 1. Compute base MI matrix
    auto mi = compute_mutual_information_matrix(pop, genome_len);

    // 2. Initial clusters: each variable alone
    std::vector<std::vector<int>> clusters;
    clusters.reserve(genome_len);
    for (int i = 0; i < genome_len; ++i)
        clusters.push_back({i});

    // Start FOS with univariate subsets
    fos = clusters;

    // 3. Agglomerative clustering:
    // repeatedly merge the pair of clusters with highest average MI.
    while (clusters.size() > 1)
    {
        double best_score = -std::numeric_limits<double>::infinity();
        int best_a = -1, best_b = -1;

        for (int a = 0; a < (int)clusters.size(); ++a)
        {
            for (int b = a + 1; b < (int)clusters.size(); ++b)
            {
                double sum = 0.0;
                int cnt = 0;

                for (int i : clusters[a])
                {
                    for (int j : clusters[b])
                    {
                        sum += mi[i][j];
                        ++cnt;
                    }
                }

                double avg = (cnt > 0) ? (sum / cnt) : 0.0;

                if (avg > best_score)
                {
                    best_score = avg;
                    best_a = a;
                    best_b = b;
                }
            }
        }

        if (best_a == -1 || best_b == -1)
            break; // degenerate case: no usable merge

        // 4. Merge best_a and best_b
        std::vector<int> merged = clusters[best_a];
        merged.insert(merged.end(),
                      clusters[best_b].begin(), clusters[best_b].end());
        std::sort(merged.begin(), merged.end());

        // Append merged cluster to FOS
        fos.push_back(merged);

        // Replace cluster a with merged, erase b
        clusters[best_a] = std::move(merged);
        clusters.erase(clusters.begin() + best_b);
    }

    return fos;
}

void gomea_step(Population &pop, const FOS &fos, const Dataset &data, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> pop_dist(0, (int)pop.size() - 1);

    // Random permutation of indices
    std::vector<int> order(pop.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    for (int idx : order)
    {
        Individual &base = pop[idx];
        Individual candidate = base;

        // Random order of subsets for this individual
        std::vector<int> fos_idx(fos.size());
        std::iota(fos_idx.begin(), fos_idx.end(), 0);
        std::shuffle(fos_idx.begin(), fos_idx.end(), rng);

        for (int fi : fos_idx)
        {
            const auto &subset = fos[fi];

            // pick a random donor different from idx
            int donor_idx = idx;
            while (donor_idx == idx)
            {
                donor_idx = pop_dist(rng);
            }
            const Individual &donor = pop[donor_idx];

            // backup
            std::vector<int> backup_genes;
            backup_genes.reserve(subset.size());
            for (int pos : subset)
            {
                backup_genes.push_back(candidate.genome[pos]);
            }

            // mix genes from donor
            for (size_t k = 0; k < subset.size(); ++k)
            {
                int pos = subset[k];
                candidate.genome[pos] = donor.genome[pos];
            }

            double new_f = evaluate_fitness(candidate.genome, data);
            if (new_f < candidate.fitness)
            {
                candidate.fitness = new_f; // accept
            }
            else
            {
                // reject -> restore
                for (size_t k = 0; k < subset.size(); ++k)
                {
                    int pos = subset[k];
                    candidate.genome[pos] = backup_genes[k];
                }
            }
        }

        // Replace if improved
        if (candidate.fitness < base.fitness)
        {
            base = std::move(candidate);
        }
    }
}
