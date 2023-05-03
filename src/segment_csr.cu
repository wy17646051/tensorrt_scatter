#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "common/reduction.cuh"
#include "common/reduction.h"
#include "common/utils.cuh"
#include "segment_csr.h"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff
#define SEGMENT_CSR_LAUNCH_INSTANTIATION_TR(T, R)                                                                      \
template                                                                                                               \
int32_t segment_csr_launch<T, R>(const T *src, const std::vector<int32_t> &src_size, const int32_t *indptr,            \
                                 const std::vector<int32_t> &indptr_size, std::tuple<T*, int32_t*> out,                \
                                 cudaStream_t stream);                                                                 \
template                                                                                                               \
int32_t segment_csr_launch<T, R>(const T *src, const std::vector<int32_t> &src_size, const int32_t *indptr,            \
                                 const std::vector<int32_t> &indptr_size, const T *base,                               \
                                 std::tuple<T*, int32_t*> out, cudaStream_t stream);
#define SEGMENT_CSR_LAUNCH_INSTANTIATION(T)                                                                            \
SEGMENT_CSR_LAUNCH_INSTANTIATION_TR(T, ReductionType::SUM)                                                             \
SEGMENT_CSR_LAUNCH_INSTANTIATION_TR(T, ReductionType::MEAN)                                                            \
SEGMENT_CSR_LAUNCH_INSTANTIATION_TR(T, ReductionType::MUL)                                                             \
SEGMENT_CSR_LAUNCH_INSTANTIATION_TR(T, ReductionType::DIV)                                                             \
SEGMENT_CSR_LAUNCH_INSTANTIATION_TR(T, ReductionType::MIN)                                                             \
SEGMENT_CSR_LAUNCH_INSTANTIATION_TR(T, ReductionType::MAX)

inline __device__ int indptr_to_offset(const int32_t *indptr_size, int32_t indptr_dim, int32_t idx) 
{
    int offset = idx % (indptr_size[indptr_dim - 1] - 1), stride = 1;
    idx /= indptr_size[indptr_dim - 1] - 1;
    for (int i = indptr_dim - 2; i >= 0; --i) {
        stride *= indptr_size[i + 1];
        offset += (idx % indptr_size[i]) * stride;
        idx /= indptr_size[i];
    }
    return offset;
}

template <typename scalar_t, ReductionType REDUCE, int TB>
__global__ void segment_csr_kernel(const scalar_t *src, const int32_t *indptr, const int32_t *indptr_size, 
                                   int32_t indptr_dim, scalar_t *out, int32_t *arg_out, size_t N, size_t E)
{
    // Each warp processes exactly `32/TB` rows and aggregates all row values
    // via a parallel reduction.

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = thread_idx / TB;
    int lane_idx = thread_idx & (TB - 1);
    if (row_idx >= N)
        return;

    int offset = indptr_to_offset(indptr_size, indptr_dim, row_idx);
    int row_start = __ldg(indptr + offset);
    int row_end = __ldg(indptr + offset + 1);

    scalar_t val = Reducer<scalar_t, REDUCE>::init();
    int arg, arg_tmp;

    offset = (row_idx / (indptr_size[indptr_dim - 1] - 1)) * E;
    for (auto src_idx = row_start + lane_idx; src_idx < row_end; src_idx += TB)
        Reducer<scalar_t, REDUCE>::update(&val, src[offset + src_idx], &arg, src_idx);

#pragma unroll
    for (int i = TB / 2; i > 0; i /= 2)
    {
        // Parallel reduction inside a single warp.
        if (REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX)
            arg_tmp =  __shfl_down_sync(FULL_MASK, arg, i);
        Reducer<scalar_t, REDUCE>::update(&val, __shfl_down_sync(FULL_MASK, val, i), &arg, arg_tmp);
    }

    if (lane_idx == 0)
        if (arg_out != nullptr)
            Reducer<scalar_t, REDUCE>::write(out + row_idx, val, arg_out + row_idx, arg, row_end - row_start);
        else
            Reducer<scalar_t, REDUCE>::write(out + row_idx, val, row_end - row_start);
}

template <typename scalar_t, ReductionType REDUCE>
__global__ void segment_csr_broadcast_kernel(const scalar_t *src, const int32_t *indptr, const int32_t *indptr_size, 
                                             int32_t indptr_dim, scalar_t *out, int32_t *arg_out, size_t N, size_t K, 
                                             size_t E)
{
    // Each thread processes exactly one row. It turned out that is more
    // efficient than using shared memory due to avoiding synchronization
    // barriers.

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = thread_idx / K;
    int lane_idx = thread_idx % K;
    if (thread_idx >= N * K)
        return;
    
    int offset = indptr_to_offset(indptr_size, indptr_dim, row_idx);
    int row_start = __ldg(indptr + offset);
    int row_end = __ldg(indptr + offset + 1);

    scalar_t val = Reducer<scalar_t, REDUCE>::init();
    int arg;

    offset = (row_idx / (indptr_size[indptr_dim - 1] - 1)) * E * K;
    for (auto src_idx = row_start; src_idx < row_end; src_idx++)
        Reducer<scalar_t, REDUCE>::update(&val, src[offset + K * src_idx + lane_idx], &arg, src_idx);

    if (arg_out != nullptr)
        Reducer<scalar_t, REDUCE>::write(out + thread_idx, val, arg_out + thread_idx, arg, row_end - row_start);
    else
        Reducer<scalar_t, REDUCE>::write(out + thread_idx, val, row_end - row_start);
}

//! \todo test different devices (cudaSetDevice(src.get_device());)
//! \todo expand index
template<typename scalar_t, ReductionType REDUCE>
int32_t segment_csr_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *indptr, 
                           const std::vector<int32_t> &indptr_size, const scalar_t *base, 
                           std::tuple<scalar_t*, int32_t*> out, cudaStream_t stream)
{
    if (src_size.size() < indptr_size.size())
        return -1;

    if (!std::equal(indptr_size.begin(), indptr_size.end() - 1, src_size.begin()))
        return -1;

    auto dim = indptr_size.size() - 1;

    auto _mul = [](int a, int b) { return a * b; };
    auto src_numel = std::accumulate(src_size.begin(), src_size.end(), 1, _mul);
    auto indptr_numel = std::accumulate(indptr_size.begin(), indptr_size.end(), 1, _mul);
    auto out_numel = src_numel / src_size[dim] * std::max<int32_t>(indptr_size[dim] - 1, 0);

    cudaMemcpyAsync(std::get<0>(out), base, sizeof(scalar_t) * out_numel, cudaMemcpyDeviceToDevice, stream);

    if ((REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX) && std::get<1>(out) != nullptr)
        fill_kernel<int32_t>
        <<<BLOCKS(1, out_numel), THREADS, 0, stream>>>(std::get<1>(out), out_numel, src_size[dim]);

    if (src_numel == 0)
        return 0;

    auto N = max(indptr_size[dim] - 1, 0) * (indptr_numel / indptr_size[dim]);
    auto K = out_numel / N;
    auto E = src_size[dim];
    int32_t *indptr_size_dev;
    cudaMallocAsync(&indptr_size_dev, sizeof(int32_t) * indptr_size.size(), stream);
    cudaMemcpyAsync(indptr_size_dev, indptr_size.data(), sizeof(int32_t) * indptr_size.size(), cudaMemcpyHostToDevice, stream);

    if (K == 1)
        segment_csr_kernel<scalar_t, REDUCE, 1>
        <<<BLOCKS(32, N), THREADS, 0, stream>>>(
            src, indptr, indptr_size_dev, indptr_size.size(), std::get<0>(out), std::get<1>(out), N, E);
    else
        segment_csr_broadcast_kernel<scalar_t, REDUCE>
        <<<BLOCKS(1, N * K), THREADS, 0, stream>>>(
            src, indptr, indptr_size_dev, indptr_size.size(), std::get<0>(out), std::get<1>(out), N, K, E);

    cudaFreeAsync(indptr_size_dev, stream);
    return 0;
}

template<typename scalar_t, ReductionType REDUCE>
int32_t segment_csr_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *indptr, 
                           const std::vector<int32_t> &indptr_size, std::tuple<scalar_t*, int32_t*> out, 
                           cudaStream_t stream)
{
    auto dim = indptr_size.size() - 1;
    auto src_numel = std::accumulate(src_size.begin(), src_size.end(), 1, [](int a, int b) { return a * b; });
    auto out_numel = src_numel / src_size[dim] * std::max<int32_t>(indptr_size[dim] - 1, 0);

    scalar_t *base;
    cudaMallocAsync(&base, sizeof(scalar_t) * out_numel, stream);
    fill_kernel<scalar_t>
    <<<BLOCKS(1, out_numel), THREADS, 0, stream>>>(base, out_numel, (scalar_t)0);

    auto status = segment_csr_launch<scalar_t, REDUCE>(src, src_size, indptr, indptr_size, base, out, stream);
    if (status != 0)
        return status;
    
    cudaFreeAsync(base, stream);
    return 0;
}

SEGMENT_CSR_LAUNCH_INSTANTIATION(half)
SEGMENT_CSR_LAUNCH_INSTANTIATION(float)

#ifdef BUILD_PTLAUNCH
int64_t segment_csr_ptlaunth(const torch::Tensor src, const torch::Tensor indptr, 
                             const torch::optional<torch::Tensor> base, const std::string& reduce, torch::Tensor out, 
                             torch::optional<torch::Tensor> arg_out)
{
    int64_t status = -1;

    std::vector<int32_t> _src_size(src.sizes().data(), src.sizes().data()+src.sizes().size());
    std::vector<int32_t> _indptr_size(indptr.sizes().data(), indptr.sizes().data()+indptr.sizes().size());

    auto indptr_int32 = indptr.to(torch::kInt32);
    auto _indptr = indptr_int32.data_ptr<int32_t>();
    
    torch::optional<torch::Tensor> arg_out_int32 = torch::nullopt;
    int32_t* _arg_out = nullptr;
    if (arg_out.has_value())
    {
        arg_out_int32 = arg_out.value().to(torch::kInt32);
        _arg_out = arg_out_int32.value().data_ptr<int32_t>();
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (src.dtype() == torch::kHalf)
    {
        auto _src = reinterpret_cast<half *>(src.data_ptr<at::Half>());
        auto _base = base != torch::nullopt ? reinterpret_cast<half *>(base.value().data_ptr<at::Half>()) : nullptr;
        auto _out = std::tuple<half *const, int32_t *const>(
            reinterpret_cast<half *>(out.data_ptr<at::Half>()), _arg_out);

        AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
            if (base != torch::nullopt)
                status = segment_csr_launch<half, REDUCE>(_src, _src_size, _indptr, _indptr_size, _base, _out, stream);
            else
                status = segment_csr_launch<half, REDUCE>(_src, _src_size, _indptr, _indptr_size, _out, stream);
        });
    }
    else if (src.dtype() == torch::kFloat)
    {
        auto _src = src.data_ptr<float>();    
        auto _base = base != torch::nullopt ? base.value().data_ptr<float>() : nullptr;
        auto _out = std::tuple<float *const, int32_t *const>(out.data_ptr<float>(), _arg_out);

        AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
            if (base != torch::nullopt)
                status = segment_csr_launch<float, REDUCE>(_src, _src_size, _indptr, _indptr_size, _base, _out, stream);
            else
                status = segment_csr_launch<float, REDUCE>(_src, _src_size, _indptr, _indptr_size, _out, stream);
        });
    }
    else
    {
        throw std::runtime_error("segment_csr: unsupported data type");
    }

    if (arg_out.has_value())
    {
        at::Tensor arg_out_int64 = arg_out_int32.value().to(torch::kInt64);
        cudaMemcpyAsync(
            arg_out.value().data_ptr<int64_t>(), arg_out_int64.data_ptr<int64_t>(), 
            sizeof(int64_t) * arg_out.value().numel(), cudaMemcpyDeviceToDevice, stream
        );
    }

    cudaStreamDestroy(stream);
    return status;
}

TORCH_LIBRARY_FRAGMENT(tensorrt_scatter, m)
{
    m.def("segment_csr_ptlaunth", segment_csr_ptlaunth);
}
#endif