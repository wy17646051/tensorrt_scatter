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

#endif // TRTS_COMMON_UTILS_H
