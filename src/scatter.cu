#include <vector>
#include <tuple>
#include <numeric>
#include <string>

#include "common/reduction.cuh"
#include "common/reduction.h"
#include "common/utils.cuh"
#include "scatter.h"

#define THREADS 256
#define BLOCKS(N) (N + THREADS - 1) / THREADS
#define SCATTER_LAUNCH_INSTANTIATION_TR(T, R)                                                                          \
template                                                                                                               \
int32_t scatter_launch<T, R>(const T *src, const std::vector<int32_t> &src_size, const int32_t *index, int32_t dim,    \
                       std::tuple<T*, int32_t*> out, const std::vector<int32_t> &out_size, cudaStream_t stream);       \
template                                                                                                               \
int32_t scatter_launch<T, R>(const T *src, const std::vector<int32_t> &src_size, const int32_t *index, int32_t dim,    \
                       const T *base, std::tuple<T*, int32_t*> out, const std::vector<int32_t> &out_size,              \
                       cudaStream_t stream);
#define SCATTER_LAUNCH_INSTANTIATION(T)                                                                                \
SCATTER_LAUNCH_INSTANTIATION_TR(T, ReductionType::SUM)                                                                 \
SCATTER_LAUNCH_INSTANTIATION_TR(T, ReductionType::MEAN)                                                                \
SCATTER_LAUNCH_INSTANTIATION_TR(T, ReductionType::MUL)                                                                 \
SCATTER_LAUNCH_INSTANTIATION_TR(T, ReductionType::DIV)                                                                 \
SCATTER_LAUNCH_INSTANTIATION_TR(T, ReductionType::MIN)                                                                 \
SCATTER_LAUNCH_INSTANTIATION_TR(T, ReductionType::MAX)

template <typename scalar_t, ReductionType REDUCE>
__global__ void scatter_kernel(const scalar_t *src, const int32_t src_numel, const int32_t *index, scalar_t *out,
                               int32_t E, int32_t K, int32_t N)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= src_numel)
        return;

    int b = thread_idx / (E * K);
    int k = thread_idx % K;
    int idx = index[thread_idx];

    Reducer<scalar_t, REDUCE>::atomic_write(out + b * N * K + idx * K + k, src[thread_idx]);
}

template <typename scalar_t>
__global__ void scatter_arg_kernel(const scalar_t *src, const int32_t src_numel, const int32_t *index,
                                   const scalar_t *out, int32_t *arg_out, int32_t E, int32_t K, int32_t N)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= src_numel)
        return;

    int b = thread_idx / (E * K);
    int e = (thread_idx / K) % E;
    int k = thread_idx % K;
    int idx = index[thread_idx];

    if (src[thread_idx] == out[b * N * K + idx * K + k])
        arg_out[b * N * K + idx * K + k] = e;
}

//! \todo src, index, base, out must be contiguous
//! \todo test different devices
//! \todo broadcast index
//! \todo mean divide by N
//! \todo half is unreliable
template <typename scalar_t, ReductionType REDUCE>
int32_t scatter_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index, int32_t dim,
                       const scalar_t *base, std::tuple<scalar_t*, int32_t*> out,
                       const std::vector<int32_t> &out_size, cudaStream_t stream)
{
    if (dim < 0)
        dim += src_size.size();

    if (src_size.size() != out_size.size())
        return -1;
    for (auto i = 0; i < src_size.size(); i++)
        if (i != dim && src_size[i] != out_size[i])
            return -1;

    auto _mul = [](int a, int b) { return a * b; };
    auto src_numel = std::accumulate(src_size.begin(), src_size.end(), 1, _mul);
    auto out_numel = std::accumulate(out_size.begin(), out_size.end(), 1, _mul);

    cudaMemcpyAsync(std::get<0>(out), base, sizeof(scalar_t) * out_numel, cudaMemcpyDeviceToDevice, stream);

    if ((REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX) && std::get<1>(out) != nullptr)
        fill_kernel<int32_t>
        <<<BLOCKS(out_numel), THREADS, 0, stream>>>(std::get<1>(out), out_numel, src_size[dim]);

    if (src_numel == 0)
        return 0;

    auto B = 1;
    for (auto i = 0; i < dim; i++)
        B *= src_size[i];
    auto E = src_size[dim];
    auto K = src_numel / (B * E);
    auto N = out_size[dim];

    scatter_kernel<scalar_t, REDUCE>
    <<<BLOCKS(src_numel), THREADS, 0, stream>>>(src, src_numel, index, std::get<0>(out), E, K, N);

    if ((REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX) && std::get<1>(out) != nullptr)
        scatter_arg_kernel<scalar_t>
        <<<BLOCKS(src_numel), THREADS, 0, stream>>>(src, src_numel, index, std::get<0>(out), std::get<1>(out), E, K, N);

    return 0;
}

template <typename scalar_t, ReductionType REDUCE>
int32_t scatter_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index, int32_t dim,
                       std::tuple<scalar_t*, int32_t*> out, const std::vector<int32_t> &out_size, cudaStream_t stream)
{
    auto out_numel = std::accumulate(out_size.begin(), out_size.end(), 1, [](int a, int b) { return a * b; });

    scalar_t *base;
    cudaMallocAsync(&base, sizeof(scalar_t) * out_numel, stream);
    fill_kernel<scalar_t>
    <<<BLOCKS(out_numel), THREADS, 0, stream>>>(base, out_numel, Reducer<scalar_t, REDUCE>::init());

    auto status = scatter_launch<scalar_t, REDUCE>(src, src_size, index, dim, base, out, out_size, stream);
    if (status != 0)
        return status;

    if (REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX)
        replace_kernel<scalar_t>
        <<<BLOCKS(out_numel), THREADS, 0, stream>>>(
            std::get<0>(out), out_numel, Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);
    
    cudaFreeAsync(base, stream);
    return 0;
}

SCATTER_LAUNCH_INSTANTIATION(half)
SCATTER_LAUNCH_INSTANTIATION(float)

