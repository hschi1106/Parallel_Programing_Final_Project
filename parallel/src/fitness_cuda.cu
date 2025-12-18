// fitness_cuda.cu
// CUDA implementation of single-program fitness evaluation over many samples.

#include "types.hpp"
#include "fitness_cuda.hpp"                 // host API (pure C++)
#include "fitness_cuda_kernels.cuh"         // cuda_runtime.h + CUDA_CHECK (CUDA-only)

#include <vector>
#include <limits>
#include <cmath>
#include <cstring>
#include <algorithm>

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

// ============= fitness_cuda.hpp =============

void gpu_eval_init(GpuEvalContext& ctx, const Dataset& data, int operand_count, int prog_len)
{
    // Reset first (so destroy can be safely called on partial init)
    ctx = GpuEvalContext{};
    ctx.host_data = &data;
    ctx.prog_len = prog_len;

    if (data.empty())
    {
        // Nothing to allocate. (evaluate_fitness_gpu will return +inf.)
        return;
    }
    if (operand_count <= 0)
    {
        std::cerr << "operand_count must be > 0\n";
        std::exit(1);
    }
    // Token set currently supports VAR_1..VAR_3 only.
    if (operand_count > 3)
    {
        std::cerr << "operand_count > 3 is not supported by current token set (VAR_1..VAR_3)\n";
        std::exit(1);
    }
    if (prog_len <= 0)
    {
        std::cerr << "prog_len must be > 0\n";
        std::exit(1);
    }

    ctx.N = (int)data.size();
    ctx.D = operand_count;

    // Pack host dataset into contiguous arrays for GPU
    std::vector<double> h_X((size_t)ctx.N * (size_t)ctx.D);
    std::vector<double> h_y((size_t)ctx.N);

    for (int i = 0; i < ctx.N; ++i)
    {
        const auto& s = data[(size_t)i];
        for (int j = 0; j < ctx.D; ++j)
            h_X[(size_t)i * (size_t)ctx.D + (size_t)j] = s.inputs[(size_t)j];
        h_y[(size_t)i] = s.output;
    }

    // Device allocations
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_X, h_X.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_y, h_y.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_prog, (size_t)ctx.prog_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_sum, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(ctx.d_X, h_X.data(), h_X.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_y, h_y.data(), h_y.size() * sizeof(double), cudaMemcpyHostToDevice));

    // Pinned host buffers (required for async memcpy nodes in graphs)
    CUDA_CHECK(cudaMallocHost((void**)&ctx.h_prog_pinned, (size_t)ctx.prog_len * sizeof(int)));
    CUDA_CHECK(cudaMallocHost((void**)&ctx.h_sum_pinned, sizeof(double)));

    // Stream for graph launches
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ctx.stream = (void*)stream;

    // Build a CUDA Graph for: H2D(prog) -> memset(sum) -> kernel -> D2H(sum)
    const int threads = 256;
    int blocks = (ctx.N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaGraph_t graph = nullptr;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    CUDA_CHECK(cudaMemcpyAsync(ctx.d_prog,
                              ctx.h_prog_pinned,
                              (size_t)ctx.prog_len * sizeof(int),
                              cudaMemcpyHostToDevice,
                              stream));

    CUDA_CHECK(cudaMemsetAsync(ctx.d_sum, 0, sizeof(double), stream));

    fitness_kernel_single_prog_kernel<<<blocks, threads, 0, stream>>>(
        ctx.d_prog, ctx.prog_len, ctx.d_X, ctx.d_y, ctx.N, ctx.D, ctx.d_sum);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(ctx.h_sum_pinned,
                              ctx.d_sum,
                              sizeof(double),
                              cudaMemcpyDeviceToHost,
                              stream));

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graphExec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));

    ctx.graph_exec = (void*)graphExec;
}


void gpu_eval_destroy(GpuEvalContext& ctx)
{
    // Destroy graph + stream first (they may reference device allocations)
    if (ctx.graph_exec)
    {
        CUDA_CHECK(cudaGraphExecDestroy((cudaGraphExec_t)ctx.graph_exec));
        ctx.graph_exec = nullptr;
    }
    if (ctx.stream)
    {
        CUDA_CHECK(cudaStreamDestroy((cudaStream_t)ctx.stream));
        ctx.stream = nullptr;
    }

    if (ctx.h_prog_pinned)
    {
        CUDA_CHECK(cudaFreeHost(ctx.h_prog_pinned));
        ctx.h_prog_pinned = nullptr;
    }
    if (ctx.h_sum_pinned)
    {
        CUDA_CHECK(cudaFreeHost(ctx.h_sum_pinned));
        ctx.h_sum_pinned = nullptr;
    }

    if (ctx.d_X)
    {
        CUDA_CHECK(cudaFree(ctx.d_X));
        ctx.d_X = nullptr;
    }
    if (ctx.d_y)
    {
        CUDA_CHECK(cudaFree(ctx.d_y));
        ctx.d_y = nullptr;
    }
    if (ctx.d_prog)
    {
        CUDA_CHECK(cudaFree(ctx.d_prog));
        ctx.d_prog = nullptr;
    }
    if (ctx.d_sum)
    {
        CUDA_CHECK(cudaFree(ctx.d_sum));
        ctx.d_sum = nullptr;
    }

    ctx = GpuEvalContext{};
}


double evaluate_fitness_gpu(GpuEvalContext& ctx, const std::vector<int>& prog)
{
    if (ctx.N == 0) return std::numeric_limits<double>::infinity();
    if ((int)prog.size() != ctx.prog_len) return 1e12;
    if (!ctx.h_prog_pinned || !ctx.h_sum_pinned || !ctx.stream || !ctx.graph_exec) return 1e12;

    // Update pinned host program buffer (graph copies from this address every launch)
    std::memcpy(ctx.h_prog_pinned, prog.data(), (size_t)ctx.prog_len * sizeof(int));

    CUDA_CHECK(cudaGraphLaunch((cudaGraphExec_t)ctx.graph_exec, (cudaStream_t)ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)ctx.stream));

    double sum = *(ctx.h_sum_pinned);
    double out = sum / (double)ctx.N;
    if (!std::isfinite(out)) return 1e12;
    return out;
}

