#ifndef TRTS_SEGMENT_CSR_H
#define TRTS_SEGMENT_CSR_H

#include <tuple>
#include <vector>
#ifdef BUILD_PTLAUNCH
#include <torch/extension.h>
#endif

#include "common/reduction.h"

template<typename scalar_t, ReductionType REDUCE>
int32_t segment_csr_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *indptr, 
                           const std::vector<int32_t> &indptr_size, const scalar_t *base, 
                           std::tuple<scalar_t*, int32_t*> out, cudaStream_t stream);

template<typename scalar_t, ReductionType REDUCE>
int32_t segment_csr_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *indptr, 
                           const std::vector<int32_t> &indptr_size, std::tuple<scalar_t*, int32_t*> out, 
                           cudaStream_t stream);

#ifdef BUILD_PTLAUNCH
//! \warning This function does not do any input legitimacy checking and is mainly used as a wrapper for testing the 
//! segment_csr_launch function on python.
int64_t segment_csr_ptlaunth(const torch::Tensor src, const torch::Tensor indptr, 
                             const torch::optional<torch::Tensor> base, const std::string& reduce, torch::Tensor out, 
                             torch::optional<torch::Tensor> arg_out);
#endif

#endif // TRTS_SEGMENT_CSR_H