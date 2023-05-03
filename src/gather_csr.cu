#include <numeric>
#include <vector>

#include "common/utils.cuh"
#include "gather_csr.h"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff
#define GATHER_CSR_LAUNCH_INSTANTIATION(T)                                                                             \
template                                                                                                               \
int32_t gather_csr_launch<T>(const T *src, const std::vector<int32_t> &src_size, const int32_t *indptr,                \
                             const std::vector<int32_t> &indptr_size, const T *base, T* out,                           \
                             const std::vector<int32_t> &out_size, cudaStream_t stream);                               \
template                                                                                                               \
int32_t gather_csr_launch<T>(const T *src, const std::vector<int32_t> &src_size, const int32_t *indptr,                \
                             const std::vector<int32_t> &indptr_size, T* out, const std::vector<int32_t> &out_size,    \
                             cudaStream_t stream);

template <typename scalar_t, int TB>
__global__ void gather_csr_kernel(const scalar_t *src, const int32_t *indptr, const int32_t *indptr_size, 
                                  int32_t indptr_dim, scalar_t *out, size_t N, size_t E)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = thread_idx / TB;
    int lane_idx = thread_idx % TB;
    if (row_idx >= N)
        return;

    int offset = indptr_to_offset(indptr_size, indptr_dim, row_idx);
    int row_start = __ldg(indptr + offset);
    int row_end = __ldg(indptr + offset + 1);
    scalar_t val = __ldg(src + row_idx);

    offset = (row_idx / (indptr_size[indptr_dim - 1] - 1)) * E;
    for (int out_idx = row_start + lane_idx; out_idx < row_end; out_idx += TB)
        out[offset + out_idx] = val; // "Mostly" coalesced.
}

template <typename scalar_t>
__global__ void gather_csr_broadcast_kernel(const scalar_t *src, const int32_t *indptr, const int32_t *indptr_size, 
                                            int32_t indptr_dim, scalar_t *out, size_t N, size_t K, size_t E)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = thread_idx / K;
    int lane_idx = thread_idx % K;
    if (thread_idx >= N * K)
        return;

    int offset = indptr_to_offset(indptr_size, indptr_dim, row_idx);
    int row_start = __ldg(indptr + offset);
    int row_end = __ldg(indptr + offset + 1);

    scalar_t val = src[thread_idx]; // Coalesced.

    offset = (row_idx / (indptr_size[indptr_dim - 1] - 1)) * E * K;
    for (int out_idx = row_start; out_idx < row_end; out_idx++)
        out[offset + K * out_idx + lane_idx] = val; // "Mostly" coalesced.
}

//! \todo test different devices (cudaSetDevice(src.get_device());)
//! \todo expand index
template<typename scalar_t>
int32_t gather_csr_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *indptr, 
                          const std::vector<int32_t> &indptr_size, const scalar_t *base, scalar_t* out, 
                          const std::vector<int32_t> &out_size, cudaStream_t stream)
{
    if (src_size.size() < indptr_size.size() || src_size.size() != out_size.size())
        return -1;

    if (!std::equal(indptr_size.begin(), indptr_size.end()-1, src_size.begin()))
        return -1;

    auto dim = indptr_size.size() - 1;

    if (src_size[dim] != 0 && src_size[dim] != indptr_size[dim] - 1)
        return -1;
    
    for (int i = 0; i < src_size.size(); i++)
        if (i != dim && src_size[i] != out_size[i])
            return -1;
    
    auto _mul = [](int a, int b) { return a * b; };
    auto src_numel = std::accumulate(src_size.begin(), src_size.end(), 1, _mul);
    auto indptr_numel = std::accumulate(indptr_size.begin(), indptr_size.end(), 1, _mul);
    auto out_numel = std::accumulate(out_size.begin(), out_size.end(), 1, _mul);

    cudaMemcpyAsync(out, base, sizeof(scalar_t) * out_numel, cudaMemcpyDeviceToDevice, stream);

    if (src_numel == 0)
        return 0;

    auto D = indptr_size[dim];
    auto N = src_size[dim] * (indptr_numel / indptr_size[dim]);
    auto K = src_numel / N;
    auto E = out_size[dim];
    int32_t *indptr_size_dev;
    cudaMallocAsync(&indptr_size_dev, sizeof(int32_t) * indptr_size.size(), stream);
    cudaMemcpyAsync(indptr_size_dev, indptr_size.data(), sizeof(int32_t) * indptr_size.size(), cudaMemcpyHostToDevice, stream);

    if (K == 1)
        gather_csr_kernel<scalar_t, 4>
        <<<BLOCKS(1, 4 * N), THREADS, 0, stream>>>(src, indptr, indptr_size_dev, indptr_size.size(), out, N, E);
    else
        gather_csr_broadcast_kernel<scalar_t>
        <<<BLOCKS(1, N * K), THREADS, 0, stream>>>(src, indptr, indptr_size_dev, indptr_size.size(), out, N, K, E);

    return 0;
}

template<typename scalar_t>
int32_t gather_csr_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *indptr, 
                          const std::vector<int32_t> &indptr_size, scalar_t* out, const std::vector<int32_t> &out_size, 
                          cudaStream_t stream)
{
    auto out_numel = std::accumulate(out_size.begin(), out_size.end(), 1, [](int a, int b) { return a * b; });

    scalar_t *base;
    cudaMallocAsync(&base, sizeof(scalar_t) * out_numel, stream);
    fill_kernel<scalar_t>
    <<<BLOCKS(1, out_numel), THREADS, 0, stream>>>(base, out_numel, (scalar_t)0);

    auto status = gather_csr_launch<scalar_t>(src, src_size, indptr, indptr_size, base, out, out_size, stream);
    if (status != 0)
        return status;
    
    cudaFreeAsync(base, stream);
    return 0;
}

GATHER_CSR_LAUNCH_INSTANTIATION(half)
GATHER_CSR_LAUNCH_INSTANTIATION(float)

#ifdef BUILD_PTLAUNCH
int64_t gather_csr_ptlaunth(const torch::Tensor src, const torch::Tensor indptr, 
                            const torch::optional<torch::Tensor> base, torch::Tensor out)
{
    int64_t status = -1;

    std::vector<int32_t> _src_size(src.sizes().data(), src.sizes().data()+src.sizes().size());
    std::vector<int32_t> _indptr_size(indptr.sizes().data(), indptr.sizes().data()+indptr.sizes().size());
    std::vector<int32_t> _out_size(out.sizes().data(), out.sizes().data()+out.sizes().size());

    auto indptr_int32 = indptr.to(torch::kInt32);
    auto _indptr = indptr_int32.data_ptr<int32_t>();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (src.dtype() == torch::kHalf)
    {
        auto _src = reinterpret_cast<half *>(src.data_ptr<at::Half>());
        auto _base = base != torch::nullopt ? reinterpret_cast<half *>(base.value().data_ptr<at::Half>()) : nullptr;
        auto _out = reinterpret_cast<half *>(out.data_ptr<at::Half>());

        if (base != torch::nullopt)
            status = gather_csr_launch<half>(_src, _src_size, _indptr, _indptr_size, _base, _out, _out_size, stream);
        else
            status = gather_csr_launch<half>(_src, _src_size, _indptr, _indptr_size, _out, _out_size, stream);
       
    }
    else if (src.dtype() == torch::kFloat)
    {
        auto _src = src.data_ptr<float>();    
        auto _base = base != torch::nullopt ? base.value().data_ptr<float>() : nullptr;
        auto _out = out.data_ptr<float>();

        if (base != torch::nullopt)
            status = gather_csr_launch<float>(_src, _src_size, _indptr, _indptr_size, _base, _out, _out_size, stream);
        else
            status = gather_csr_launch<float>(_src, _src_size, _indptr, _indptr_size, _out, _out_size, stream);
    }
    else
    {
        throw std::runtime_error("gather_csr: unsupported data type");
    }

    cudaStreamDestroy(stream);
    return status;
}

TORCH_LIBRARY_FRAGMENT(tensorrt_scatter, m)
{
    m.def("gather_csr_ptlaunth", gather_csr_ptlaunth);
}
#endif