#include "gpg_cuda.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Config
#define MAX_STACK 32       // Match serial code stack reserve
#define BLOCK_SIZE 128     // Threads per block
#define PENALTY_FITNESS 1e6 // Match serial code PENALTY

// Globals
static float total_mi_time = 0.0f;
static float total_gomea_time = 0.0f;
cudaEvent_t start, stop;

// Device pointers
int *d_genomes = nullptr;
double *d_fitnesses = nullptr;
double *d_data_inputs = nullptr; // Flattened [sample0_var0, s0_v1, ..., s1_v0...]
double *d_data_outputs = nullptr;
curandState *d_rng_states = nullptr;
double *d_mi_matrix = nullptr;

int *d_fos_data = nullptr;
int *d_fos_offsets = nullptr;
int *d_fos_sizes = nullptr;

// Constants
int g_num_samples = 0;
int g_num_inputs = 0; // Number of variables per sample

void init_timers() {
    cudaEventCreate(&start); cudaEventCreate(&stop);
    total_mi_time = 0.0f; total_gomea_time = 0.0f;
}
void destroy_timers() {
    cudaEventDestroy(start); cudaEventDestroy(stop);
}
void gpu_print_profile_stats() {
    std::cout << "GPU MI Time: " << total_mi_time << " ms, GOMEA Time: " << total_gomea_time << " ms\n";
}

// ========== Device Helpers ==========

__device__ double block_reduce_sum(double val) {
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

// Stack Machine Evaluation (Shared Genome)
// inputs_ptr 指向當前 Sample 的輸入起始位置
__device__ double eval_program_shared(const int* __restrict__ s_genome, int len, const double* inputs_ptr) {
    double stack[MAX_STACK];
    int sp = 0;

    for (int i = 0; i < len; ++i) {
        int tok = s_genome[i];
        
        if (tok >= VAR_1) { // Variable
            int var_idx = tok - VAR_1;
            // 假設 g_num_inputs 正確，且 prog 合法
            stack[sp++] = inputs_ptr[var_idx]; 
        } else {
            // Unary or Binary
            if (tok == OP_SIN || tok == OP_COS || tok == OP_EXP) {
                if (sp < 1) return PENALTY_FITNESS;
                double a = stack[--sp];
                double r = 0.0;
                if (tok == OP_SIN) r = sin(a);
                else if (tok == OP_COS) r = cos(a);
                else { // EXP Protected
                    if (a <= 10.0) r = exp(a);
                    else r = exp(10.0);
                }
                if (!isfinite(r)) return PENALTY_FITNESS;
                stack[sp++] = r;
            }
            else { // Binary
                if (sp < 2) return PENALTY_FITNESS;
                double b = stack[--sp];
                double a = stack[--sp];
                double r = 0.0;
                if (tok == OP_ADD) r = a + b;
                else if (tok == OP_SUB) r = a - b;
                else if (tok == OP_MUL) r = a * b;
                else { // DIV Protected (Match Serial: |b| >= 0.001)
                    if (fabs(b) >= 0.001) r = a / b;
                    else r = 1.0;
                }
                if (!isfinite(r)) return PENALTY_FITNESS;
                stack[sp++] = r;
            }
        }
    }
    if (sp != 1) return PENALTY_FITNESS;
    return isfinite(stack[0]) ? stack[0] : PENALTY_FITNESS;
}

// ========== Kernels ==========

__global__ void init_rng_kernel(curandState* states, unsigned long seed, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void compute_mi_kernel(const int* genomes, double* mi_matrix, int pop_size, int genome_len) {
    int i = blockIdx.x; int j = blockIdx.y;
    if (i >= genome_len || j >= genome_len || j <= i) {
        if (i==j && i<genome_len) mi_matrix[i*genome_len+j] = 0.0;
        return;
    }
    // MAX_TOKENS_SUPPORTED defined in types (approx 20)
    const int ALPHABET = 20; 
    __shared__ int sh_counts_ij[ALPHABET][ALPHABET];
    __shared__ int sh_counts_i[ALPHABET];
    __shared__ int sh_counts_j[ALPHABET];
    
    int tid = threadIdx.x;
    if (tid < ALPHABET) {
        sh_counts_i[tid] = 0; sh_counts_j[tid] = 0;
        for(int k=0; k<ALPHABET; ++k) sh_counts_ij[tid][k] = 0;
    }
    __syncthreads();

    for (int p = tid; p < pop_size; p += blockDim.x) {
        int ti = genomes[p * genome_len + i];
        int tj = genomes[p * genome_len + j];
        // Clamp to avoid shared mem overflow if random gen is weird
        if(ti < ALPHABET && tj < ALPHABET && ti >= 0 && tj >= 0) {
            atomicAdd(&sh_counts_i[ti], 1);
            atomicAdd(&sh_counts_j[tj], 1);
            atomicAdd(&sh_counts_ij[ti][tj], 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        double mi = 0.0; double N = (double)pop_size;
        for (int a = 0; a < ALPHABET; ++a) {
            double pi = sh_counts_i[a] / N;
            if (pi <= 0) continue;
            for (int b = 0; b < ALPHABET; ++b) {
                double pj = sh_counts_j[b] / N;
                int c_ij = sh_counts_ij[a][b];
                if (pj <= 0 || c_ij == 0) continue;
                double pij = c_ij / N;
                mi += pij * log(pij / (pi * pj));
            }
        }
        mi_matrix[i * genome_len + j] = mi; mi_matrix[j * genome_len + i] = mi;
    }
}

// Block Parallel GOMEA
// Grid: POP_SIZE, Block: BLOCK_SIZE
__global__ void gomea_block_parallel_kernel(
    int* d_pop_genomes, double* d_pop_fitnesses,
    const double* d_data_inputs, const double* d_data_outputs,
    const int* d_fos_data, const int* d_fos_offsets, const int* d_fos_sizes, int num_subsets,
    int pop_size, int genome_len, int n_samples, int n_inputs,
    curandState* rng_states
) {
    int pid = blockIdx.x; // Individual
    int tid = threadIdx.x; 

    if (pid >= pop_size) return;

    extern __shared__ int s_genome[];
    int* s_backup = &s_genome[genome_len];

    // Load Genome
    for (int k = tid; k < genome_len; k += blockDim.x) {
        s_genome[k] = d_pop_genomes[pid * genome_len + k];
    }
    
    __shared__ double current_fitness;
    __shared__ curandState local_rng;
    if (tid == 0) {
        current_fitness = d_pop_fitnesses[pid];
        local_rng = rng_states[pid];
    }
    __syncthreads();

    // Shuffle FOS
    __shared__ int fos_indices[64]; 
    int fos_count = (num_subsets > 64) ? 64 : num_subsets;
    if (tid == 0) {
        for(int k=0; k<fos_count; ++k) fos_indices[k] = k;
        for(int k=fos_count-1; k>0; k--) {
            int swap_i = curand(&local_rng) % (k+1);
            int tmp = fos_indices[k];
            fos_indices[k] = fos_indices[swap_i];
            fos_indices[swap_i] = tmp;
        }
    }
    __syncthreads();

    // Loop Subsets
    for (int i = 0; i < fos_count; ++i) {
        int subset_idx = fos_indices[i];
        int f_start = d_fos_offsets[subset_idx];
        int f_size = d_fos_sizes[subset_idx];

        __shared__ int donor_idx;
        if (tid == 0) donor_idx = curand(&local_rng) % pop_size;
        __syncthreads();

        // Mix Genes
        if (tid == 0) {
            for(int k=0; k<f_size; ++k) {
                int gene_pos = d_fos_data[f_start + k];
                s_backup[gene_pos] = s_genome[gene_pos];
                s_genome[gene_pos] = d_pop_genomes[donor_idx * genome_len + gene_pos];
            }
        }
        __syncthreads();

        // Evaluate (Loop stride for n_samples > blockDim)
        double my_sq_err = 0.0;
        for (int k = tid; k < n_samples; k += blockDim.x) {
            const double* inputs = &d_data_inputs[k * n_inputs];
            double y = d_data_outputs[k];
            double pred = eval_program_shared(s_genome, genome_len, inputs);
            
            if (pred >= PENALTY_FITNESS) {
                my_sq_err = PENALTY_FITNESS * 10.0; // Ensure it dominates
                break;
            } else {
                double err = pred - y;
                my_sq_err += err * err;
            }
        }

        double total_err_sum = block_reduce_sum(my_sq_err);
        
        // Check for penalty saturation in sum
        double new_fitness;
        if (total_err_sum >= PENALTY_FITNESS) new_fitness = 1e12; 
        else new_fitness = total_err_sum / (double)n_samples; // MSE

        if (tid == 0) {
            if (new_fitness <= current_fitness) {
                current_fitness = new_fitness;
            } else {
                // Revert
                for(int k=0; k<f_size; ++k) {
                    int gene_pos = d_fos_data[f_start + k];
                    s_genome[gene_pos] = s_backup[gene_pos];
                }
            }
        }
        __syncthreads();
    }

    // Write back
    for (int k = tid; k < genome_len; k += blockDim.x) {
        d_pop_genomes[pid * genome_len + k] = s_genome[k];
    }
    if (tid == 0) {
        d_pop_fitnesses[pid] = current_fitness;
        rng_states[pid] = local_rng;
    }
}

// ========== Host Functions ==========

void gpu_init(const Dataset &data, int pop_size, int genome_len, unsigned int seed) {
    init_timers();
    g_num_samples = data.size();
    if (g_num_samples == 0) return;
    g_num_inputs = data[0].inputs.size();

    // Flatten Data
    std::vector<double> h_inputs;
    std::vector<double> h_outputs;
    h_inputs.reserve(g_num_samples * g_num_inputs);
    h_outputs.reserve(g_num_samples);

    for (const auto& s : data) {
        h_inputs.insert(h_inputs.end(), s.inputs.begin(), s.inputs.end());
        h_outputs.push_back(s.output);
    }

    cudaMalloc(&d_data_inputs, h_inputs.size() * sizeof(double));
    cudaMalloc(&d_data_outputs, h_outputs.size() * sizeof(double));
    cudaMemcpy(d_data_inputs, h_inputs.data(), h_inputs.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_outputs, h_outputs.data(), h_outputs.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_genomes, pop_size * genome_len * sizeof(int));
    cudaMalloc(&d_fitnesses, pop_size * sizeof(double));
    cudaMalloc(&d_rng_states, pop_size * sizeof(curandState));
    cudaMalloc(&d_mi_matrix, genome_len * genome_len * sizeof(double));

    int max_nodes = genome_len * genome_len;
    cudaMalloc(&d_fos_data, max_nodes * sizeof(int));
    cudaMalloc(&d_fos_offsets, max_nodes * sizeof(int));
    cudaMalloc(&d_fos_sizes, max_nodes * sizeof(int));

    int threads = 128;
    int blocks = (pop_size + threads - 1) / threads;
    init_rng_kernel<<<blocks, threads>>>(d_rng_states, seed, pop_size);
    cudaDeviceSynchronize();
}

void gpu_free() {
    cudaFree(d_genomes); cudaFree(d_fitnesses);
    cudaFree(d_data_inputs); cudaFree(d_data_outputs);
    cudaFree(d_rng_states); cudaFree(d_mi_matrix);
    cudaFree(d_fos_data); cudaFree(d_fos_offsets); cudaFree(d_fos_sizes);
    destroy_timers();
}

void gpu_load_population(const Population &pop) {
    if (pop.empty()) return;
    std::vector<int> flat;
    std::vector<double> fits;
    for (const auto& ind : pop) {
        flat.insert(flat.end(), ind.genome.begin(), ind.genome.end());
        fits.push_back(ind.fitness);
    }
    cudaMemcpy(d_genomes, flat.data(), flat.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fitnesses, fits.data(), fits.size() * sizeof(double), cudaMemcpyHostToDevice);
}

void gpu_retrieve_population(Population &pop) {
    int pop_size = pop.size();
    int len = pop[0].genome.size();
    std::vector<int> flat(pop_size * len);
    std::vector<double> fits(pop_size);
    cudaMemcpy(flat.data(), d_genomes, flat.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fits.data(), d_fitnesses, fits.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < pop_size; ++i) {
        pop[i].genome.assign(flat.begin() + i*len, flat.begin() + (i+1)*len);
        pop[i].fitness = fits[i];
    }
}

void gpu_retrieve_fitnesses(std::vector<double> &fitnesses) {
    cudaMemcpy(fitnesses.data(), d_fitnesses, fitnesses.size() * sizeof(double), cudaMemcpyDeviceToHost);
}

std::vector<double> gpu_compute_mi(int pop_size, int genome_len) {
    cudaEventRecord(start);
    dim3 grid(genome_len, genome_len);
    compute_mi_kernel<<<grid, 128>>>(d_genomes, d_mi_matrix, pop_size, genome_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    total_mi_time += ms;
    std::vector<double> mi(genome_len * genome_len);
    cudaMemcpy(mi.data(), d_mi_matrix, mi.size() * sizeof(double), cudaMemcpyDeviceToHost);
    return mi;
}

void gpu_run_gomea_generation_block_parallel(const FOS &fos, int pop_size, int genome_len) {
    std::vector<int> flat_data;
    std::vector<int> offsets;
    std::vector<int> sizes;
    for(const auto& s : fos) {
        offsets.push_back(flat_data.size());
        sizes.push_back(s.size());
        flat_data.insert(flat_data.end(), s.begin(), s.end());
    }
    cudaMemcpy(d_fos_data, flat_data.data(), flat_data.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fos_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fos_sizes, sizes.data(), sizes.size() * sizeof(int), cudaMemcpyHostToDevice);

    int threads = BLOCK_SIZE;
    size_t shared_mem = 2 * genome_len * sizeof(int);
    
    cudaEventRecord(start);
    gomea_block_parallel_kernel<<<pop_size, threads, shared_mem>>>(
        d_genomes, d_fitnesses, d_data_inputs, d_data_outputs,
        d_fos_data, d_fos_offsets, d_fos_sizes, (int)fos.size(),
        pop_size, genome_len, g_num_samples, g_num_inputs, d_rng_states
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    total_gomea_time += ms;
}