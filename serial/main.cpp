#include <bits/stdc++.h>

// Simplified GP-GOMEA baseline (CPU only)
// - Representation: fixed-length postfix program (sequence of integer tokens)
// - Problem: 1D symbolic regression (y ≈ f(x))
// - GOMEA: univariate FOS (each gene = one token position)
// WARNING: Toy implementation for experiments / GPU porting, not production-quality GP.

struct Sample
{
    std::vector<double> inputs;
    double output;
};

using Dataset = std::vector<Sample>;

enum Token : int
{
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2,
    OP_DIV = 3,
    OP_SIN = 4,
    OP_COS = 5,
    OP_EXP = 6,
    VAR_1 = 7,
    VAR_2 = 8,
    VAR_3 = 9,
    TOKEN_MIN = 0,
    TOKEN_MAX = 9
};

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

struct Individual
{
    std::vector<int> genome; // token sequence (postfix)
    double fitness = std::numeric_limits<double>::infinity();
};

using Population = std::vector<Individual>;
using FOS = std::vector<std::vector<int>>; // Family of Subsets

// Evaluate one program on one sample using a simple stack-based VM.
// If the program is invalid (stack underflow, wrong final stack size, NaN),
// we return a large penalty.
static double eval_program_single(const std::vector<int> &prog, const std::vector<double>& inputs)
{
    static const double PENALTY = 1e6;
    std::vector<double> stack;
    stack.reserve(32);

    for (int tok : prog)
    {
        switch (tok)
        {
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
            else if (tok == OP_DIV)
            {
                // protected division
                if (std::fabs(b) >= 0.001)
                    r = a / b;
                else
                    r = 1;
            }
            if (!std::isfinite(r))
                return PENALTY;
            stack.push_back(r);
            break;
        }
        case OP_SIN:
        case OP_COS:
        case OP_EXP:
        {
            if (stack.empty())
                return PENALTY;
            stack.pop_back();
            double a = stack.back();
            // stack.pop_back();
            double r = 0.0;
            if (tok == OP_SIN)
                r = std::sin(a);
            else if (tok == OP_COS)
                r = std::cos(a);
            else if (tok == OP_EXP)
            {
                if (a <= 10.0)
                    r = std::exp(a);
                else
                    r = std::exp(10.0); // protect against overflow 
            }
            if (!std::isfinite(r))
                return PENALTY;
            stack.push_back(r);
            break;
        }
        case VAR_1:
        case VAR_2:
        case VAR_3:
        {
            int var_idx = tok - VAR_1;
            if (var_idx < 0 || var_idx >= (int)inputs.size())
                return PENALTY;
            stack.push_back(inputs[var_idx]);
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
        double y_hat = eval_program_single(prog, s.inputs);
        double diff = y_hat - s.output;
        sum += diff * diff;
        if (!std::isfinite(sum))
            return 1e12;
    }
    return sum / static_cast<double>(data.size());
}

// Generate a syntactically valid postfix program with only binary operators.
// genome_len must be odd.
static std::vector<int> random_program(int genome_len, std::mt19937 &rng, int num_inputs)
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
            std::uniform_int_distribution<int> op_dist(VAR_1, VAR_1 + num_inputs - 1);
            int tok = op_dist(rng);
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
    ind.genome = random_program(genome_len, rng, (int)(data.front().inputs.size()));
    ind.fitness = evaluate_fitness(ind.genome, data);
    return ind;
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


int main(int argc, char **argv)
{
    // Hyperparameters
    const int POP_SIZE = 4096;
    const int GENOME_LEN = 31; // must be odd
    const int MAX_GENERATIONS = 20;

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
    if (!fin_train) return 1;

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
    if (!fin_test) return 1;
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

        gomea_step(pop, fos, train_data, rng);
        best_it = get_best();
        std::cout << "Gen " << gen + 1 << ": best fitness = " << best_it->fitness << '\n';
    }

    // Evaluate best on test set
    double test_fitness = evaluate_fitness(best_it->genome, test_data);
    std::cout << "Best test fitness: " << test_fitness << '\n';
    // Print final program
    std::cout << "Best program (postfix): "
              << program_to_postfix_string(best_it->genome) << "\n";
    // Print input program
    std::cout << "Best program (infix):   "
              << program_to_infix_string(best_it->genome) << "\n";

    return 0;
}