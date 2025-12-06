#include <bits/stdc++.h>

// Simplified GP-GOMEA baseline (CPU only)
// - Representation: fixed-length postfix program (sequence of integer tokens)
// - Problem: 1D symbolic regression (y ≈ f(x))
// - GOMEA: univariate FOS (each gene = one token position)
// WARNING: Toy implementation for experiments / GPU porting, not production-quality GP.

struct Sample
{
    double x;
    double y;
};

using Dataset = std::vector<Sample>;

// Token encoding
// 0: variable x
// 1: constant 1.0
// 2: constant 2.0
// 3: ADD
// 4: SUB
// 5: MUL
// 6: DIV (protected)

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

static std::string token_to_string(int tok)
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

static std::string program_to_postfix_string(const std::vector<int> &prog)
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

static std::string program_to_infix_string(const std::vector<int> &prog)
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
            // operand: x, 1, 2 ...
            st.push_back(token_to_string(tok));
        }
        else
        {
            // binary operator: need two operands on stack
            if (st.size() < 2)
            {
                return "<invalid postfix program>";
            }
            std::string rhs = std::move(st.back());
            st.pop_back();
            std::string lhs = std::move(st.back());
            st.pop_back();

            std::string expr = "(" + lhs + " " + op_to_string(tok) + " " + rhs + ")";
            st.push_back(std::move(expr));
        }
    }

    if (st.size() != 1)
    {
        return "<invalid postfix program>";
    }
    return st.back();
}

struct Individual
{
    std::vector<int> genome; // token sequence (postfix)
    double fitness = std::numeric_limits<double>::infinity();
};

using Population = std::vector<Individual>;
using FOS = std::vector<std::vector<int>>; // Family of Subsets

static void print_FOS(const FOS &fos)
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

// Evaluate one program on one sample using a simple stack-based VM.
// If the program is invalid (stack underflow, wrong final stack size, NaN),
// we return a large penalty.
static double eval_program_single(const std::vector<int> &prog, double x)
{
    static const double PENALTY = 1e6;
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

// Fitness = sum of squared errors over dataset. Lower is better.
static double evaluate_fitness(const std::vector<int> &prog, const Dataset &data)
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

// Generate a syntactically valid postfix program with only binary operators.
// genome_len must be odd.
static std::vector<int> random_program(int genome_len, std::mt19937 &rng)
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

// Create a random *syntactically valid* postfix program with only binary operators.
// Requirement: genome_len should be odd (L = 2 * num_funcs + 1).
static Individual random_individual(int genome_len, std::mt19937 &rng, const Dataset &data)
{
    Individual ind;
    ind.genome = random_program(genome_len, rng);
    ind.fitness = evaluate_fitness(ind.genome, data);
    return ind;
}

// Univariate FOS: each position alone
static FOS make_univariate_fos(int genome_len)
{
    FOS fos;
    fos.reserve(genome_len);
    for (int i = 0; i < genome_len; ++i)
    {
        fos.push_back({i});
    }
    return fos;
}

// Compute pairwise mutual information between genome positions
// using discrete token values in the current population.
static std::vector<std::vector<double>>
compute_mutual_information_matrix(const Population &pop, int genome_len)
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

// Build a linkage-tree-style FOS from the current population,
// using mutual information as similarity between variables.
static FOS build_linkage_tree_fos(const Population &pop, int genome_len)
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

// One GOMEA generation using given FOS.
static void gomea_step(Population &pop, const FOS &fos, const Dataset &data, std::mt19937 &rng)
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

static Dataset make_synthetic_dataset(int n_samples, const std::vector<int> &target_prog, std::mt19937 &rng)
{
    Dataset data;
    data.reserve(n_samples);
    std::uniform_real_distribution<double> dist_x(-2.0, 2.0);
    std::normal_distribution<double> noise(0.0, 0.1); // you can set noise=0 if you want

    for (int i = 0; i < n_samples; ++i)
    {
        double x = dist_x(rng);
        double y_clean = eval_program_single(target_prog, x);
        double y = y_clean + noise(rng);
        data.push_back({x, y});
    }
    return data;
}

int main()
{
    // Hyperparameters
    const int POP_SIZE = 4096;
    const int GENOME_LEN = 31; // must be odd
    const int TARGET_LEN = 31; // target function postfix length
    const int N_SAMPLES = 128;
    const int MAX_GENERATIONS = 1000;

    // Fixed seed for reproducibility
    const unsigned SEED = 123456u;
    std::mt19937 rng(SEED);

    // 1. Generate random target program
    std::vector<int> target_prog = random_program(TARGET_LEN, rng);

    // 2. Print target program
    std::cout << "Target program (postfix): "
              << program_to_postfix_string(target_prog) << "\n";

    std::cout << "Target program (infix):   "
              << program_to_infix_string(target_prog) << "\n";

    // 3. Build dataset using target_prog
    Dataset data = make_synthetic_dataset(N_SAMPLES, target_prog, rng);

    // 4. Initialize population
    Population pop;
    pop.reserve(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i)
    {
        pop.push_back(random_individual(GENOME_LEN, rng, data));
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
        // Rebuild FOS from current population using MI-based linkage tree
        FOS fos = build_linkage_tree_fos(pop, actual_genome_len);
        // print_FOS(fos); // uncomment to debug FOS

        gomea_step(pop, fos, data, rng);
        best_it = get_best();
        std::cout << "Gen " << gen + 1 << ": best fitness = " << best_it->fitness << '\n';
    }

    std::cout << "Done. Final best fitness: " << best_it->fitness << '\n';

    // Print final program
    std::cout << "Best program (postfix): "
              << program_to_postfix_string(best_it->genome) << "\n";

    std::cout << "Best program (infix):   "
              << program_to_infix_string(best_it->genome) << "\n";

    return 0;
}
