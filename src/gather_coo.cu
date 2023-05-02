#include <string>
#include <vector>

#include "common/utils.cuh"
#include "gather_coo.h"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff
#define GATHER_COO_LAUNCH_INSTANTIATION(T)                                                                             \
template                                                                                                               \
int32_t gather_coo_launch<T>(const T *src, const std::vector<int32_t> &src_size, const int32_t *index,                 \
                             const std::vector<int32_t> &index_size, T *out, cudaStream_t stream);                     \
template                                                                                                               \
int32_t gather_coo_launch<T>(const T *src, const std::vector<int32_t> &src_size, const int32_t *index,                 \
                             const std::vector<int32_t> &index_size, const T *base, T* out, cudaStream_t stream);

template <typename scalar_t>
__global__ void gather_coo_kernel(const scalar_t *src, const int32_t *index, int32_t index_numel, scalar_t *out, 
                                  size_t N, size_t D)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= index_numel)
        return;
    
    int row = index[row_idx];
    int offset = (row_idx / D) * N;
    scalar_t val = __ldg(src + offset + row);

    out[row_idx] = val;
}

template <typename scalar_t>
__global__ void gather_coo_broadcast_kernel(const scalar_t *src, const int32_t *index, int32_t index_numel, 
                                            scalar_t *out, size_t K, size_t N, size_t D)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = thread_idx / K;
    int col_idx = thread_idx % K;
    if (thread_idx >= index_numel * K)
        return;

    int row = index[row_idx];
    int offset = (row_idx / D) * N * K;
    scalar_t val = __ldg(src + offset + K * row + col_idx);

    out[thread_idx] = val;
}

//! \todo test different devices (cudaSetDevice(src.get_device());)
//! \todo expand index
template<typename scalar_t>
int32_t gather_coo_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index,
                          const std::vector<int32_t> &index_size, const scalar_t *base, scalar_t* out, 
                          cudaStream_t stream)
{
    if (src_size.size() < index_size.size())
        return -1;

    if (!std::equal(index_size.begin(), index_size.end()-1, src_size.begin()))
        return -1;

    auto dim = index_size.size() - 1;
    
    auto _mul = [](int a, int b) { return a * b; };
    auto src_numel = std::accumulate(src_size.begin(), src_size.end(), 1, _mul);
    auto index_numel = std::accumulate(index_size.begin(), index_size.end(), 1, _mul);
    auto out_numel = src_numel / src_size[dim] * index_size[dim];

    cudaMemcpyAsync(out, base, sizeof(scalar_t) * out_numel, cudaMemcpyDeviceToDevice, stream);

    if (index_numel == 0)
        return 0;
    
    auto D = index_size[index_size.size()-1];
    auto K = out_numel / index_numel;
    auto N = src_size[dim];

    if (K == 1)
        gather_coo_kernel<scalar_t>
        <<<BLOCKS(1, index_numel), THREADS, 0, stream>>>(src, index, index_numel, out, N, D);
    else
        gather_coo_broadcast_kernel<scalar_t>
        <<<BLOCKS(1, index_numel * K), THREADS, 0, stream>>>(src, index, index_numel, out, K, N, D);
    
    return 0;
}

template<typename scalar_t>
int32_t gather_coo_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index,
                          const std::vector<int32_t> &index_size, scalar_t* out, cudaStream_t stream)
{
    auto dim = index_size.size() - 1;
    auto out_numel = std::accumulate(src_size.begin(), src_size.end(), 1,[](int a, int b) { return a * b; }) \
        / src_size[dim] * index_size[dim];

    scalar_t *base;
    cudaMallocAsync(&base, sizeof(scalar_t) * out_numel, stream);

    auto status = gather_coo_launch<scalar_t>(src, src_size, index, index_size, base, out, stream);
    if (status != 0)
        return status;
    
    cudaFreeAsync(base, stream);
    return 0;
}

GATHER_COO_LAUNCH_INSTANTIATION(half)
GATHER_COO_LAUNCH_INSTANTIATION(float)

#ifdef BUILD_PTLAUNCH
int64_t gather_coo_ptlaunth(const torch::Tensor src, const torch::Tensor index, 
                            const torch::optional<torch::Tensor> base, torch::Tensor out)
{
    int64_t status = -1;

    std::vector<int32_t> _src_size(src.sizes().data(), src.sizes().data()+src.sizes().size());
    std::vector<int32_t> _index_size(index.sizes().data(), index.sizes().data()+index.sizes().size());

    auto index_int32 = index.to(torch::kInt32);
    auto _index = index_int32.data_ptr<int32_t>();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (src.dtype() == torch::kHalf)
    {
        auto _src = reinterpret_cast<half *>(src.data_ptr<at::Half>());
        auto _base = base != torch::nullopt ? reinterpret_cast<half *>(base.value().data_ptr<at::Half>()) : nullptr;
        auto _out = reinterpret_cast<half *>(out.data_ptr<at::Half>());

        if (base != torch::nullopt)
            status = gather_coo_launch<half>(_src, _src_size, _index, _index_size, _base, _out, stream);
        else
            status = gather_coo_launch<half>(_src, _src_size, _index, _index_size, _out, stream);
       
    }
    else if (src.dtype() == torch::kFloat)
    {
        auto _src = src.data_ptr<float>();    
        auto _base = base != torch::nullopt ? base.value().data_ptr<float>() : nullptr;
        auto _out = out.data_ptr<float>();

        if (base != torch::nullopt)
            status = gather_coo_launch<float>(_src, _src_size, _index, _index_size, _base, _out, stream);
        else
            status = gather_coo_launch<float>(_src, _src_size, _index, _index_size, _out, stream);
    }
    else
    {
        throw std::runtime_error("gather_coo: unsupported data type");
    }

    cudaStreamDestroy(stream);
    return status;
}

TORCH_LIBRARY_FRAGMENT(tensorrt_scatter, m)
{
    m.def("gather_coo_ptlaunth", gather_coo_ptlaunth);
}
#endif