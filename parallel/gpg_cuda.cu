// gpg_cuda.cu
#include "gpg_cuda.hpp"
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <vector>

// ==========================================
// CONFIG: Profiling (Accumulated)
// ==========================================
// 累計時間變數 (ms)
static float total_mi_time = 0.0f;
static float total_gomea_time = 0.0f;

cudaEvent_t start, stop;

void init_timers() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    total_mi_time = 0.0f;
    total_gomea_time = 0.0f;
}

void destroy_timers() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 輔助函式：印出統計結果
void gpu_print_profile_stats() {
    std::cout << "\n=========================================\n";
    std::cout << " GPU Profiling Summary\n";
    std::cout << "=========================================\n";
    std::cout << std::left << std::setw(25) << "Total MI Computation" << ": " << total_mi_time << " ms\n";
    std::cout << std::left << std::setw(25) << "Total GOMEA Generation" << ": " << total_gomea_time << " ms\n";
    std::cout << "=========================================\n";
}

// ==========================================
// GPU CONSTANTS & GLOBALS
// ==========================================
#define MAX_STACK 32
#define PENALTY_FITNESS 1e9

int *d_genomes = nullptr;
double *d_fitnesses = nullptr;
double *d_data_x = nullptr;
double *d_data_y = nullptr;
curandState *d_rng_states = nullptr;

double *d_mi_matrix = nullptr;
int *d_fos_data = nullptr;
int *d_fos_offsets = nullptr;
int *d_fos_sizes = nullptr;

// ==========================================
// DEVICE FUNCTIONS
// ==========================================
__device__ double eval_program_device(const int* prog, int len, double x) {
    double stack[MAX_STACK];
    int sp = 0;

    for (int i = 0; i < len; ++i) {
        int tok = prog[i];
        if (tok <= 2) { 
            double val;
            if (tok == 0) val = x;
            else if (tok == 1) val = 1.0;
            else val = 2.0;
            if (sp < MAX_STACK) stack[sp++] = val;
        } else { 
            if (sp < 2) return PENALTY_FITNESS;
            double b = stack[--sp];
            double a = stack[--sp];
            double r = 0.0;
            
            if (tok == 3) r = a + b;      
            else if (tok == 4) r = a - b; 
            else if (tok == 5) r = a * b; 
            else {                        
                if (fabs(b) < 1e-9) r = a; 
                else r = a / b;
            }
            if (!isfinite(r)) return PENALTY_FITNESS;
            stack[sp++] = r;
        }
    }
    if (sp < 1) return PENALTY_FITNESS;
    return isfinite(stack[sp-1]) ? stack[sp-1] : PENALTY_FITNESS;
}

__device__ double calculate_fitness_device(const int* genome, int len, const double* data_x, const double* data_y, int n_samples) {
    double sum_sq_err = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        double pred = eval_program_device(genome, len, data_x[i]);
        if (pred >= PENALTY_FITNESS) return PENALTY_FITNESS;
        double err = pred - data_y[i];
        sum_sq_err += err * err;
    }
    return sum_sq_err;
}

// ==========================================
// KERNELS
// ==========================================
__global__ void init_rng_kernel(curandState* states, unsigned long seed, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void compute_mi_kernel(const int* genomes, double* mi_matrix, int pop_size, int genome_len) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= genome_len || j >= genome_len) return;
    if (j <= i) { 
        if (i == j) mi_matrix[i * genome_len + j] = 0.0;
        return;
    }

    const int ALPHABET = 7; 
    __shared__ int sh_counts_ij[ALPHABET][ALPHABET];
    __shared__ int sh_counts_i[ALPHABET];
    __shared__ int sh_counts_j[ALPHABET];

    int tid = threadIdx.x;
    if (tid < ALPHABET) {
        sh_counts_i[tid] = 0;
        sh_counts_j[tid] = 0;
        for(int k=0; k<ALPHABET; ++k) sh_counts_ij[tid][k] = 0;
    }
    __syncthreads();

    for (int p = tid; p < pop_size; p += blockDim.x) {
        int ti = genomes[p * genome_len + i];
        int tj = genomes[p * genome_len + j];
        atomicAdd(&sh_counts_i[ti], 1);
        atomicAdd(&sh_counts_j[tj], 1);
        atomicAdd(&sh_counts_ij[ti][tj], 1);
    }
    __syncthreads();

    if (tid == 0) {
        double mi = 0.0;
        double N = (double)pop_size;
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
        mi_matrix[i * genome_len + j] = mi;
        mi_matrix[j * genome_len + i] = mi; 
    }
}

__global__ void gomea_generation_kernel(
    int* genomes, double* fitnesses, 
    const double* data_x, const double* data_y,
    const int* fos_data, const int* fos_offsets, const int* fos_sizes, int num_subsets,
    int pop_size, int genome_len, int n_samples, curandState* rng_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_rng = rng_states[idx];
    
    int current_genome[64];
    for(int k=0; k<genome_len; ++k) current_genome[k] = genomes[idx * genome_len + k];
    
    double current_fitness = fitnesses[idx];
    int backup_genome[64];

    // Randomize FOS order
    int fos_indices[64];
    int count = (num_subsets > 64) ? 64 : num_subsets;
    for(int k=0; k<count; ++k) fos_indices[k] = k;

    for(int k=count-1; k>0; k--) {
        int swap_i = curand(&local_rng) % (k+1);
        int tmp = fos_indices[k];
        fos_indices[k] = fos_indices[swap_i];
        fos_indices[swap_i] = tmp;
    }

    for (int i = 0; i < count; ++i) {
        int subset_idx = fos_indices[i];
        int start = fos_offsets[subset_idx];
        int size = fos_sizes[subset_idx];

        int donor_idx = curand(&local_rng) % pop_size;
        
        for(int k=0; k<size; ++k) {
            int gene_pos = fos_data[start + k];
            backup_genome[gene_pos] = current_genome[gene_pos];
            current_genome[gene_pos] = genomes[donor_idx * genome_len + gene_pos];
        }

        double new_fitness = calculate_fitness_device(current_genome, genome_len, data_x, data_y, n_samples);

        if (new_fitness <= current_fitness) {
            current_fitness = new_fitness;
        } else {
            for(int k=0; k<size; ++k) {
                int gene_pos = fos_data[start + k];
                current_genome[gene_pos] = backup_genome[gene_pos];
            }
        }
    }

    for(int k=0; k<genome_len; ++k) genomes[idx * genome_len + k] = current_genome[k];
    fitnesses[idx] = current_fitness;
    rng_states[idx] = local_rng;
}

__global__ void eval_all_kernel(int* genomes, double* fits, double* dx, double* dy, int pop, int len, int samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop) {
        fits[idx] = calculate_fitness_device(&genomes[idx*len], len, dx, dy, samples);
    }
}

// ==========================================
// HOST FUNCTIONS
// ==========================================

void gpu_init(const Dataset &data, int pop_size, int genome_len, unsigned int seed) {
    init_timers(); // 初始化計時器

    std::vector<double> h_x, h_y;
    h_x.reserve(data.size());
    h_y.reserve(data.size());
    for (const auto& sample : data) {
        h_x.push_back(sample.x);
        h_y.push_back(sample.y);
    }

    size_t bytes_data = data.size() * sizeof(double);
    cudaMalloc(&d_data_x, bytes_data);
    cudaMalloc(&d_data_y, bytes_data);
    cudaMemcpy(d_data_x, h_x.data(), bytes_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_y, h_y.data(), bytes_data, cudaMemcpyHostToDevice);

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
    cudaFree(d_data_x); cudaFree(d_data_y);
    cudaFree(d_rng_states);
    cudaFree(d_mi_matrix);
    cudaFree(d_fos_data); cudaFree(d_fos_offsets); cudaFree(d_fos_sizes);
    destroy_timers(); // 清除計時器
}

// ... gpu_load_population, gpu_retrieve_population, gpu_retrieve_fitnesses 不變 ...
// (為節省篇幅，這部分與上一版相同，請保留)
void gpu_load_population(const Population &pop) {
    if (pop.empty()) return;
    int pop_size = pop.size();
    int len = pop[0].genome.size();
    std::vector<int> flat;
    flat.reserve(pop_size * len);
    for (const auto& ind : pop) {
        flat.insert(flat.end(), ind.genome.begin(), ind.genome.end());
    }
    cudaMemcpy(d_genomes, flat.data(), flat.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    int threads = 128;
    int blocks = (pop_size + threads - 1) / threads;
    eval_all_kernel<<<blocks, threads>>>(d_genomes, d_fitnesses, d_data_x, d_data_y, pop_size, len, 128);
    cudaDeviceSynchronize();
}

void gpu_retrieve_population(Population &pop) {
    int pop_size = pop.size();
    int len = pop[0].genome.size();
    std::vector<int> flat(pop_size * len);
    std::vector<double> fits(pop_size);

    cudaMemcpy(flat.data(), d_genomes, flat.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fits.data(), d_fitnesses, fits.size() * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < pop_size; ++i) {
        for (int k = 0; k < len; ++k) {
            pop[i].genome[k] = flat[i * len + k];
        }
        pop[i].fitness = fits[i];
    }
}

void gpu_retrieve_fitnesses(std::vector<double> &fitnesses) {
    cudaMemcpy(fitnesses.data(), d_fitnesses, fitnesses.size() * sizeof(double), cudaMemcpyDeviceToHost);
}

std::vector<double> gpu_compute_mi(int pop_size, int genome_len) {
    cudaEventRecord(start); // 開始計時

    dim3 grid(genome_len, genome_len);
    compute_mi_kernel<<<grid, 256>>>(d_genomes, d_mi_matrix, pop_size, genome_len);
    cudaDeviceSynchronize();

    cudaEventRecord(stop); // 停止計時
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    total_mi_time += ms; // 累加

    std::vector<double> mi(genome_len * genome_len);
    cudaMemcpy(mi.data(), d_mi_matrix, mi.size() * sizeof(double), cudaMemcpyDeviceToHost);
    return mi;
}

void gpu_run_gomea_generation(const FOS &fos, int pop_size, int genome_len, int n_samples) {
    // 1. Flatten FOS
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

    // 2. Launch Kernel
    int threads = 128;
    int blocks = (pop_size + threads - 1) / threads;
    
    cudaEventRecord(start); // 開始計時

    gomea_generation_kernel<<<blocks, threads>>>(
        d_genomes, d_fitnesses, d_data_x, d_data_y,
        d_fos_data, d_fos_offsets, d_fos_sizes, (int)fos.size(),
        pop_size, genome_len, n_samples, d_rng_states
    );
    cudaDeviceSynchronize();

    cudaEventRecord(stop); // 停止計時
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    total_gomea_time += ms; // 累加
}

double evaluate_fitness_gpu(const std::vector<int> &prog, const Dataset &data) {
    return 0.0;
}