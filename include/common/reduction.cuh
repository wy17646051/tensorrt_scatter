#ifndef TRTS_COMMON_REDUCTION_CUH
#define TRTS_COMMON_REDUCTION_CUH

#include <limits>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "common/atomics.cuh"
#include "common/reduction.h"

template <typename scalar_t, ReductionType REDUCE>
struct Reducer
{
    static inline __host__ __device__ scalar_t init()
    {
        if (REDUCE == MUL || REDUCE == DIV)
            return (scalar_t)1;
        else if (REDUCE == MIN)
            if (std::is_same<scalar_t, __half>::value || std::is_same<scalar_t, __nv_bfloat16>::value)
            {
                scalar_t max_val;
                *(reinterpret_cast<unsigned short *>(&(max_val))) =
                    std::is_same<scalar_t, __half>::value ? 0x7C00 : 0x7F80;
                return max_val;
            }
            else
                return std::numeric_limits<scalar_t>::max();
        else if (REDUCE == MAX)
            if (std::is_same<scalar_t, __half>::value || std::is_same<scalar_t, __nv_bfloat16>::value)
            {
                scalar_t min_val;
                *(reinterpret_cast<unsigned short *>(&(min_val))) =
                    std::is_same<scalar_t, __half>::value ? 0xFC00 : 0xFF80;
                return min_val;
            }
            else
                return std::numeric_limits<scalar_t>::lowest();
        else
            return (scalar_t)0;
    }

    static inline __host__ __device__ void update(scalar_t *val, scalar_t new_val)
    {
        if (REDUCE == SUM || REDUCE == MEAN)
            *val = *val + new_val;
        else if (REDUCE == MUL)
            *val = *val * new_val;
        else if (REDUCE == DIV)
            *val = *val / new_val;
        else if ((REDUCE == MIN && new_val < *val) || (REDUCE == MAX && new_val > *val))
            *val = new_val;
    }

    static inline __host__ __device__ void update(scalar_t *val, scalar_t new_val, int32_t *arg, int32_t new_arg)
    {
        if (REDUCE == SUM || REDUCE == MEAN)
            *val = *val + new_val;
        else if (REDUCE == MUL)
            *val = *val * new_val;
        else if (REDUCE == DIV)
            *val = *val / new_val;
        else if ((REDUCE == MIN && new_val < *val) || (REDUCE == MAX && new_val > *val))
        {
            *val = new_val;
            *arg = new_arg;
        }
    }

    static inline __host__ __device__ void write(scalar_t *address, scalar_t val, int count)
    {
        if (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV)
            *address = val;
        else if (REDUCE == MEAN)
            *address = val / (scalar_t)(count > 0 ? count : 1);
        else if (REDUCE == MIN || REDUCE == MAX)
        {
            if (count > 0)
                *address = val;
            else
                *address = (scalar_t)0;
        }
    }

    static inline __host__ __device__ void write(scalar_t *address, scalar_t val, int32_t *arg_address, int32_t arg,
                                                 int count)
    {
        if (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV)
            *address = val;
        else if (REDUCE == MEAN)
            *address = val / (scalar_t)(count > 0 ? count : 1);
        else if (REDUCE == MIN || REDUCE == MAX)
        {
            if (count > 0)
            {
                *address = val;
                *arg_address = arg;
            }
            else
                *address = (scalar_t)0;
        }
    }

    static inline __device__ void atomic_write(scalar_t *address, scalar_t val)
    {
        if (REDUCE == SUM || REDUCE == MEAN)
            atomAdd(address, val);
        else if (REDUCE == MUL)
            atomMul(address, val);
        else if (REDUCE == DIV)
            atomDiv(address, val);
        else if (REDUCE == MIN)
            atomMin(address, val);
        else if (REDUCE == MAX)
            atomMax(address, val);
    }
};

#endif // TRTS_COMMON_REDUCTION_CUH