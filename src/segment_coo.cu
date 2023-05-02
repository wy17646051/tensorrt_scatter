#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "common/reduction.cuh"
#include "common/reduction.h"
#include "common/utils.cuh"
#include "segment_coo.h"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff
#define SEGMENT_COO_LAUNCH_INSTANTIATION_TR(T, R)                                                                      \
template                                                                                                               \
int32_t segment_coo_launch<T, R>(const T *src, const std::vector<int32_t> &src_size, const int32_t *index,             \
                                 const std::vector<int32_t> &index_size, std::tuple<T*, int32_t*> out,                 \
                                 const std::vector<int32_t> &out_size, cudaStream_t stream);                           \
template                                                                                                               \
int32_t segment_coo_launch<T, R>(const T *src, const std::vector<int32_t> &src_size, const int32_t *index,             \
                                 const std::vector<int32_t> &index_size, const T *base, std::tuple<T*, int32_t*> out,  \
                                 const std::vector<int32_t> &out_size, cudaStream_t stream);
#define SEGMENT_COO_LAUNCH_INSTANTIATION(T)                                                                            \
SEGMENT_COO_LAUNCH_INSTANTIATION_TR(T, ReductionType::SUM)                                                             \
SEGMENT_COO_LAUNCH_INSTANTIATION_TR(T, ReductionType::MEAN)                                                            \
SEGMENT_COO_LAUNCH_INSTANTIATION_TR(T, ReductionType::MUL)                                                             \
SEGMENT_COO_LAUNCH_INSTANTIATION_TR(T, ReductionType::DIV)                                                             \
SEGMENT_COO_LAUNCH_INSTANTIATION_TR(T, ReductionType::MIN)                                                             \
SEGMENT_COO_LAUNCH_INSTANTIATION_TR(T, ReductionType::MAX)

template <typename scalar_t, ReductionType REDUCE>
__global__ void segment_coo_kernel(const scalar_t *src, const int32_t *index, int32_t index_numel, scalar_t *out, 
                                   size_t N, size_t D)
{
    // Each thread processes exactly one entry. Within a warp, we perform a
    // parallel reduction across equal indices, and write the intermediate
    // result via atomics.

    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_idx = row_idx & (32 - 1);
    if (row_idx >= index_numel)
        return;

    int idx = index[row_idx], next_idx;
    int out_idx = (row_idx / D) * N + idx;

    scalar_t reduce_val = src != nullptr ? src[row_idx]: (scalar_t)1, tmp;

#pragma unroll
    for (int i = 1; i < 32; i *= 2) {
        // Parallel reduction inside a single warp.
        tmp = __shfl_up_sync(FULL_MASK, reduce_val, i);
        next_idx = __shfl_up_sync(FULL_MASK, idx, i);
        if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
            assert(idx >= next_idx);
            if (idx == next_idx)
                Reducer<scalar_t, REDUCE>::update(&reduce_val, tmp);
        }
    }

    next_idx = __shfl_down_sync(FULL_MASK, idx, 1);
    if (lane_idx == 32 - 1 || row_idx / D != (row_idx + 1) / D || idx != next_idx)
        Reducer<scalar_t, REDUCE>::atomic_write(out + out_idx, reduce_val);
}

template <typename scalar_t>
__global__ void segment_coo_arg_kernel(const scalar_t *src, const int32_t *index, int32_t index_numel, 
                                       const scalar_t *out, int32_t *arg_out, size_t N, size_t D) 
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= index_numel)
        return;

    int idx = index[row_idx];
    int out_idx = (row_idx / D) * N + idx;

    scalar_t val = __ldg(out + out_idx);
    if (src[row_idx] == val)
        arg_out[out_idx] = row_idx % D;
}

template <typename scalar_t, ReductionType REDUCE, int TB>
__global__ void segment_coo_broadcast_kernel(const scalar_t *src, const int32_t *index, int32_t index_numel,
                                             scalar_t *out, size_t K, size_t N, size_t D)
{
    // Each thread processes a single column and `TB` index entries. Coalesced
    // read and write is performed in column-major order. The intermediate
    // results are written via atomics.
    int E_1 = index_numel / D;
    int E_2 = (D - 1) + TB - ((D - 1) % TB);

    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.y * blockDim.x + threadIdx.x;

    int dim_start = (row_idx * TB) / E_2;
    int row_start = (row_idx * TB) % E_2;

    if (dim_start >= E_1 || col_idx >= K)
        return;
    
    int idx1 = __ldg(index + dim_start * D + row_start), idx2;
    scalar_t reduce_val = src != nullptr ? src[K * (dim_start * D + row_start) + col_idx] : (scalar_t)1, new_reduce_val;

#pragma unroll
    for (int i = 1; i < TB; i++) {
        if (row_start + i >= D)
            break;

        idx2 = __ldg(index + dim_start * D + row_start + i);
        new_reduce_val = src != nullptr ? src[K * (dim_start * D + row_start + i) + col_idx] : (scalar_t)1;
        assert(idx1 <= idx2);

        if (idx1 == idx2) {
            Reducer<scalar_t, REDUCE>::update(&reduce_val, new_reduce_val);
        } else {
            Reducer<scalar_t, REDUCE>::atomic_write(out + (dim_start * N + idx1) * K + col_idx, reduce_val);
            reduce_val = new_reduce_val;
        }
        idx1 = idx2;
    }

    Reducer<scalar_t, REDUCE>::atomic_write(out + (dim_start * N + idx1) * K + col_idx, reduce_val);
}

template <typename scalar_t>
__global__ void segment_coo_arg_broadcast_kernel(const scalar_t *src, const int32_t *index, int32_t index_numel, 
                                                 const scalar_t *out, int32_t *arg_out, size_t K, size_t N, size_t D) 
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = thread_idx / K;
    int col_idx = thread_idx % K;
    if (row_idx >= index_numel || col_idx >= K)
        return;

    int idx = __ldg(index + row_idx);
    int out_idx = ((row_idx / D) * N + idx) * K + col_idx;

    scalar_t val = __ldg(out + out_idx);
    if (src[thread_idx] == val)
        arg_out[out_idx] = row_idx % D;
}

#define SEGMENT_COO_KERNEL(scalar_t, REDUCE, AVG_LEN, E_1, E_2, K, N, src, index, index_numel, out, stream)            \
if (K == 1)                                                                                                            \
    segment_coo_kernel<scalar_t, REDUCE>                                                                               \
    <<<BLOCKS(1, index_numel), THREADS, 0, stream>>>(src, index, index_numel, out, N, E_2);                            \
else if (AVG_LEN <= 8)                                                                                                 \
    segment_coo_broadcast_kernel<scalar_t, REDUCE, 4>                                                                  \
    <<<dim3((E_1 * ((E_2 + 3) / 4) + 7) / 8, (K + 31) / 32), dim3(32, 8), 0, stream>>>(                                \
        src, index, index_numel, out, K, N, E_2);                                                                      \
else if (AVG_LEN <= 16)                                                                                                \
    segment_coo_broadcast_kernel<scalar_t, REDUCE, 8>                                                                  \
    <<<dim3((E_1 * ((E_2 + 7) / 8) + 7) / 8, (K + 31) / 32), dim3(32, 8), 0, stream>>>(                                \
        src, index, index_numel, out, K, N, E_2);                                                                      \
else if (AVG_LEN <= 32)                                                                                                \
    segment_coo_broadcast_kernel<scalar_t, REDUCE, 16>                                                                 \
    <<<dim3((E_1 * ((E_2 + 15) / 16) + 7) / 8, (K + 31) / 32), dim3(32, 8), 0, stream>>>(                              \
        src, index, index_numel, out, K, N, E_2);                                                                      \
else                                                                                                                   \
    segment_coo_broadcast_kernel<scalar_t, REDUCE, 32>                                                                 \
    <<<dim3((E_1 * ((E_2 + 31) / 32) + 7) / 8, (K + 31) / 32), dim3(32, 8), 0, stream>>>(                              \
        src, index, index_numel, out, K, N, E_2);

//! \todo test different devices (cudaSetDevice(src.get_device());)
//! \todo expand index
template<typename scalar_t, ReductionType REDUCE>
int32_t segment_coo_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index,
                           const std::vector<int32_t> &index_size, const scalar_t *base, 
                           std::tuple<scalar_t*, int32_t*> out, const std::vector<int32_t> &out_size, 
                           cudaStream_t stream)
{
    if (src_size.size() < index_size.size() || src_size.size() != out_size.size())
        return -1;

    if (!std::equal(index_size.begin(), index_size.end(), src_size.begin()))
        return -1;

    auto dim = index_size.size() - 1;

    for (int i = 0; i < src_size.size(); i++)
        if (i != dim && src_size[i] != out_size[i])
            return -1;

    auto _mul = [](int a, int b) { return a * b; };
    auto src_numel = std::accumulate(src_size.begin(), src_size.end(), 1, _mul);
    auto index_numel = std::accumulate(index_size.begin(), index_size.end(), 1, _mul);
    auto out_numel = std::accumulate(out_size.begin(), out_size.end(), 1, _mul);

    cudaMemcpyAsync(std::get<0>(out), base, sizeof(scalar_t) * out_numel, cudaMemcpyDeviceToDevice, stream);

    if ((REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX) && std::get<1>(out) != nullptr)
        fill_kernel<int32_t>
        <<<BLOCKS(1, out_numel), THREADS, 0, stream>>>(std::get<1>(out), out_numel, src_size[dim]);

    if (index_numel == 0)
        return 0;

    auto E_2 = index_size[dim];
    auto E_1 = index_numel / E_2;
    auto K = src_numel / index_numel;
    auto N = out_size[dim];
    auto avg_len = (float)E_2 / (float)N;

    SEGMENT_COO_KERNEL(scalar_t, REDUCE, avg_len, E_1, E_2, K, N, src, index, index_numel, std::get<0>(out), stream);

    if ((REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX) && std::get<1>(out) != nullptr)
    {
        if (K == 1)
            segment_coo_arg_kernel<scalar_t>
            <<<BLOCKS(1, index_numel), THREADS, 0, stream>>>(
                src, index, index_numel, std::get<0>(out), std::get<1>(out), N, E_2);
        else
            segment_coo_arg_broadcast_kernel<scalar_t>
            <<<BLOCKS(1, index_numel * K), THREADS, 0, stream>>>(
                src, index, index_numel, std::get<0>(out), std::get<1>(out), K, N, E_2);
    }
    if (REDUCE == ReductionType::MEAN)
    {   
        scalar_t *count;
        cudaMallocAsync(&count, sizeof(scalar_t) * out_numel, stream);
        
        fill_kernel<scalar_t>
        <<<BLOCKS(1, out_numel), THREADS, 0, stream>>>(count, out_numel, (scalar_t)0);
        SEGMENT_COO_KERNEL(
            scalar_t, ReductionType::SUM, avg_len, E_1, E_2, K, N, nullptr, index, index_numel, count, stream);
        
        div_kernel<scalar_t>
        <<<BLOCKS(1, out_numel), THREADS, 0, stream>>>(std::get<0>(out), out_numel, count, true);

        cudaFreeAsync(count, stream);
    }
    return 0;
}

template<typename scalar_t, ReductionType REDUCE>
int32_t segment_coo_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index,
                           const std::vector<int32_t> &index_size, std::tuple<scalar_t*, int32_t*> out, 
                           const std::vector<int32_t> &out_size, cudaStream_t stream)
{
    auto out_numel = std::accumulate(out_size.begin(), out_size.end(), 1, [](int a, int b) { return a * b; });

    scalar_t *base;
    cudaMallocAsync(&base, sizeof(scalar_t) * out_numel, stream);
    fill_kernel<scalar_t>
    <<<BLOCKS(1, out_numel), THREADS, 0, stream>>>(base, out_numel, Reducer<scalar_t, REDUCE>::init());

    auto status = segment_coo_launch<scalar_t, REDUCE>(src, src_size, index, index_size, base, out, out_size, stream);
    if (status != 0)
        return status;

    if (REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX)
        replace_kernel<scalar_t>
        <<<BLOCKS(1, out_numel), THREADS, 0, stream>>>(
            std::get<0>(out), out_numel, Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);
    
    cudaFreeAsync(base, stream);
    return 0;
}

SEGMENT_COO_LAUNCH_INSTANTIATION(half)
SEGMENT_COO_LAUNCH_INSTANTIATION(float)

#ifdef BUILD_PTLAUNCH
int64_t segment_coo_ptlaunth(const torch::Tensor src, const torch::Tensor index, 
                             const torch::optional<torch::Tensor> base, const std::string& reduce, torch::Tensor out, 
                             torch::optional<torch::Tensor> arg_out)
{
    int64_t status = -1;

    std::vector<int32_t> _src_size(src.sizes().data(), src.sizes().data()+src.sizes().size());
    std::vector<int32_t> _index_size(index.sizes().data(), index.sizes().data()+index.sizes().size());
    std::vector<int32_t> _out_size(out.sizes().data(), out.sizes().data() + out.sizes().size());

    auto index_int32 = index.to(torch::kInt32);
    auto _index = index_int32.data_ptr<int32_t>();
    
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
                status = segment_coo_launch<half, REDUCE>(
                    _src, _src_size, _index, _index_size, _base, _out, _out_size, stream);
            else
                status = segment_coo_launch<half, REDUCE>(
                    _src, _src_size, _index, _index_size, _out, _out_size, stream);
        });
    }
    else if (src.dtype() == torch::kFloat)
    {
        auto _src = src.data_ptr<float>();    
        auto _base = base != torch::nullopt ? base.value().data_ptr<float>() : nullptr;
        auto _out = std::tuple<float *const, int32_t *const>(out.data_ptr<float>(), _arg_out);

        AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
            if (base != torch::nullopt)
                status = segment_coo_launch<float, REDUCE>(
                    _src, _src_size, _index, _index_size, _base, _out, _out_size, stream);
            else
                status = segment_coo_launch<float, REDUCE>(
                    _src, _src_size, _index, _index_size, _out, _out_size, stream);
        });
    }
    else
    {
        throw std::runtime_error("segment_coo: unsupported data type");
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
    m.def("segment_coo_ptlaunth", segment_coo_ptlaunth);
}
#endif