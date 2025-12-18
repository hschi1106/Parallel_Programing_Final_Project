// fitness_cuda.cu
// CUDA implementation of single-program fitness evaluation over many samples.

#include "types.hpp"
#include "fitness_cuda.hpp"                 // host API (pure C++)
#include "fitness_cuda_kernels.cuh"         // cuda_runtime.h + CUDA_CHECK (CUDA-only)

#include <vector>
#include <limits>
#include <cmath>

// ============= fitness_cuda_kernels.cuh =============

__device__ __forceinline__ bool finite_dev(double x) { return isfinite(x); }

// Device VM: mirrors CPU semantics (penalty / protected ops).
__device__ double eval_program_single_dev(const int* prog, int prog_len,
                                         const double* inputs, int input_dim)
{
    const double PENALTY = 1e6;
    double stack[32];
    int sp = 0;

    for (int i = 0; i < prog_len; ++i)
    {
        int tok = prog[i];
        switch (tok)
        {
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        {
            if (sp < 2) return PENALTY;
            double b = stack[--sp];
            double a = stack[--sp];
            double r = 0.0;

            if (tok == OP_ADD) r = a + b;
            else if (tok == OP_SUB) r = a - b;
            else if (tok == OP_MUL) r = a * b;
            else {
                // protected division
                r = (fabs(b) >= 0.001) ? (a / b) : 1.0;
            }

            if (!finite_dev(r)) return PENALTY;
            if (sp >= 32) return PENALTY;
            stack[sp++] = r;
            break;
        }
        case OP_SIN:
        case OP_COS:
        case OP_EXP:
        {
            if (sp < 1) return PENALTY;
            double a = stack[--sp];
            double r = 0.0;

            if (tok == OP_SIN) r = sin(a);
            else if (tok == OP_COS) r = cos(a);
            else {
                // exp cap
                r = (a <= 10.0) ? exp(a) : exp(10.0);
            }

            if (!finite_dev(r)) return PENALTY;
            if (sp >= 32) return PENALTY;
            stack[sp++] = r;
            break;
        }
        case VAR_1:
        case VAR_2:
        case VAR_3:
        {
            int var_idx = tok - VAR_1;
            if (var_idx < 0 || var_idx >= input_dim) return PENALTY;
            if (sp >= 32) return PENALTY;
            stack[sp++] = inputs[var_idx];
            break;
        }
        default:
            return PENALTY;
        }
    }

    if (sp != 1) return PENALTY;
    double v = stack[0];
    if (!finite_dev(v)) return PENALTY;
    return v;
}

__global__ void fitness_kernel_single_prog_kernel(const int* d_prog, int prog_len,
                                                  const double* d_X, const double* d_y,
                                                  int N, int D, double* d_sum_out)
{
    __shared__ double sh[256];
    int tid = threadIdx.x;

    double local = 0.0;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for (int s = idx; s < N; s += stride)
    {
        const double* x = d_X + (size_t)s * (size_t)D;
        double y_hat = eval_program_single_dev(d_prog, prog_len, x, D);
        double diff = y_hat - d_y[s];
        local += diff * diff;

        // mimic CPU behavior: bail if accumulation turns non-finite
        if (!finite_dev(local)) { local = 1e12; break; }
    }

    sh[tid] = local;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1)
    {
        if (tid < off) sh[tid] += sh[tid + off];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(d_sum_out, sh[0]);
}

// ===================== Batched evaluation kernel =====================
// grid.x = program index (0..batch-1)
// grid.y = "slice" index over samples (tiles), to increase parallelism for large N.
// Each block reduces its local sum and atomicAdds into d_sums_out[p].
__global__ void fitness_kernel_batch_kernel(const int *d_progs, int prog_len,
                                            const double *d_X, const double *d_y,
                                            int N, int D, int batch, double *d_sums_out)
{
    int p = (int)blockIdx.x;
    if (p >= batch) return;

    __shared__ double sh[256];
    int tid = (int)threadIdx.x;

    const int blocks_y = (int)gridDim.y;
    const int by = (int)blockIdx.y;

    const int *prog = d_progs + (size_t)p * (size_t)prog_len;

    double local = 0.0;
    // Iterate over samples assigned to this (p, by, tid)
    for (int s = by * (int)blockDim.x + tid; s < N; s += (int)blockDim.x * blocks_y)
    {
        const double *x = d_X + (size_t)s * (size_t)D;
        double y_hat = eval_program_single_dev(prog, prog_len, x, D);
        double diff = y_hat - d_y[s];
        local += diff * diff;

        if (!finite_dev(local)) { local = 1e12; break; }
    }

    sh[tid] = local;
    __syncthreads();

    for (int off = (int)blockDim.x / 2; off > 0; off >>= 1)
    {
        if (tid < off) sh[tid] += sh[tid + off];
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(&d_sums_out[p], sh[0]);
    }
}


// ============= fitness_cuda.hpp =============

void gpu_eval_init(GpuEvalContext& ctx, const Dataset& data, int operand_count, int prog_len)
{
    if (data.empty())
    {
        ctx = GpuEvalContext{};
        ctx.host_data = &data;
        ctx.prog_len = prog_len;
        return;
    }
    if (operand_count <= 0)
    {
        std::cerr << "operand_count must be > 0\n";
        std::exit(1);
    }
    if (operand_count > 3)
    {
        std::cerr << "operand_count=" << operand_count
                  << " but tokens only support VAR_1..VAR_3\n";
        std::exit(1);
    }

    ctx.host_data = &data;
    ctx.N = (int)data.size();
    ctx.D = operand_count;
    ctx.prog_len = prog_len;

    std::vector<double> hX((size_t)ctx.N * (size_t)ctx.D);
    std::vector<double> hy((size_t)ctx.N);

    for (int i = 0; i < ctx.N; ++i)
    {
        for (int j = 0; j < ctx.D; ++j)
            hX[(size_t)i * (size_t)ctx.D + (size_t)j] = data[i].inputs[j];
        hy[i] = data[i].output;
    }

    CUDA_CHECK(cudaMalloc(&ctx.d_X, hX.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_y, hy.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(ctx.d_X, hX.data(), hX.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_y, hy.data(), hy.size() * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&ctx.d_prog, (size_t)ctx.prog_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_sum, sizeof(double)));
}

void gpu_eval_destroy(GpuEvalContext& ctx)
{
    if (ctx.d_X) CUDA_CHECK(cudaFree(ctx.d_X));
    if (ctx.d_y) CUDA_CHECK(cudaFree(ctx.d_y));
    if (ctx.d_prog) CUDA_CHECK(cudaFree(ctx.d_prog));
    if (ctx.d_sum) CUDA_CHECK(cudaFree(ctx.d_sum));
    if (ctx.d_progs_batch) CUDA_CHECK(cudaFree(ctx.d_progs_batch));
    if (ctx.d_sums_batch) CUDA_CHECK(cudaFree(ctx.d_sums_batch));

    ctx = GpuEvalContext{};
}

double evaluate_fitness_gpu(GpuEvalContext& ctx, const std::vector<int>& prog)
{
    if (ctx.N == 0) return std::numeric_limits<double>::infinity();
    if ((int)prog.size() != ctx.prog_len) return 1e12;

    CUDA_CHECK(cudaMemcpy(ctx.d_prog, prog.data(), (size_t)prog.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx.d_sum, 0, sizeof(double)));

    const int threads = 256;
    int blocks = (ctx.N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    fitness_kernel_single_prog_kernel<<<blocks, threads>>>(
        ctx.d_prog, ctx.prog_len, ctx.d_X, ctx.d_y, ctx.N, ctx.D, ctx.d_sum);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&sum, ctx.d_sum, sizeof(double), cudaMemcpyDeviceToHost));

    double out = sum / (double)ctx.N;
    if (!std::isfinite(out)) return 1e12;
    return out;
}


static void ensure_batch_buffers(GpuEvalContext &ctx, int batch)
{
    if (batch <= 0) return;
    if (ctx.batch_cap >= batch && ctx.d_progs_batch && ctx.d_sums_batch) return;

    // grow to next power of two for fewer reallocs
    int cap = 1;
    while (cap < batch) cap <<= 1;

    if (ctx.d_progs_batch) CUDA_CHECK(cudaFree(ctx.d_progs_batch));
    if (ctx.d_sums_batch) CUDA_CHECK(cudaFree(ctx.d_sums_batch));

    CUDA_CHECK(cudaMalloc(&ctx.d_progs_batch, (size_t)cap * (size_t)ctx.prog_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_sums_batch, (size_t)cap * sizeof(double)));
    ctx.batch_cap = cap;
}

// Evaluate multiple programs (flattened) in a single kernel launch.
// Output is mean squared error (sum/N) per program.
void evaluate_fitness_gpu_batch(GpuEvalContext &ctx, const int *h_progs_flat, int batch, double *h_out)
{
    if (batch <= 0) return;
    if (ctx.N == 0) {
        for (int i = 0; i < batch; ++i) h_out[i] = std::numeric_limits<double>::infinity();
        return;
    }
    if (!h_progs_flat || !h_out) return;

    ensure_batch_buffers(ctx, batch);

    const size_t bytes = (size_t)batch * (size_t)ctx.prog_len * sizeof(int);
    CUDA_CHECK(cudaMemcpy(ctx.d_progs_batch, h_progs_flat, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx.d_sums_batch, 0, (size_t)batch * sizeof(double)));

    const int threads = 256;
    int blocks_y = (ctx.N + threads - 1) / threads;
    if (blocks_y < 1) blocks_y = 1;
    if (blocks_y > 64) blocks_y = 64; // cap to avoid ridiculous grid.y

    dim3 block(threads);
    dim3 grid((unsigned)batch, (unsigned)blocks_y);

    fitness_kernel_batch_kernel<<<grid, block>>>(
        ctx.d_progs_batch, ctx.prog_len, ctx.d_X, ctx.d_y, ctx.N, ctx.D, batch, ctx.d_sums_batch);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> sums((size_t)batch);
    CUDA_CHECK(cudaMemcpy(sums.data(), ctx.d_sums_batch, (size_t)batch * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < batch; ++i)
    {
        double out = sums[(size_t)i] / (double)ctx.N;
        if (!std::isfinite(out)) out = 1e12;
        h_out[i] = out;
    }
}
