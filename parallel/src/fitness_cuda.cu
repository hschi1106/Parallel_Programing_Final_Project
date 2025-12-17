// fitness_cuda.cu
// CUDA implementation of single-program fitness evaluation over many samples.

#include "types.hpp"
#include "fitness_cuda.hpp"                 // host API (pure C++)
#include "fitness_cuda_kernels.cuh"         // cuda_runtime.h + CUDA_CHECK (CUDA-only)

#include <vector>
#include <limits>
#include <cmath>

namespace {

__device__ __forceinline__ bool finite_dev(double x) { return isfinite(x); }

__device__ __forceinline__ double atomicAdd_double(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(addr, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

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

    if (tid == 0) atomicAdd_double(d_sum_out, sh[0]);
}

} // namespace

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
