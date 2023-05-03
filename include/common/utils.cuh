#ifndef TRTS_COMMON_UTILS_H
#define TRTS_COMMON_UTILS_H

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename scalar_t>
__global__ void fill_kernel(scalar_t *src, int32_t numel, const scalar_t val)
{
    int32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= numel)
        return;

    src[thread_idx] = val;
}

template <typename scalar_t>
__global__ void replace_kernel(scalar_t *src, int32_t numel, const scalar_t ori_val, const scalar_t new_val)
{
    int32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= numel)
        return;

    if (src[thread_idx] == ori_val)
        src[thread_idx] = new_val;
}

template <typename scalar_t>
__global__ void div_kernel(scalar_t *src, int32_t numel, const scalar_t *divisor, bool skip_zero_divisor=false)
{
    int32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= numel)
        return;

    if (skip_zero_divisor && divisor[thread_idx] == (scalar_t)0)
        return;

    src[thread_idx] /= divisor[thread_idx];
}

inline __host__ __device__ int indptr_to_offset(const int32_t *indptr_size, int32_t indptr_dim, int32_t idx) 
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

#endif // TRTS_COMMON_UTILS_H
