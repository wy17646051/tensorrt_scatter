#ifndef TRTS_SEGMENT_COO_H
#define TRTS_SEGMENT_COO_H

#include <vector>
#include <tuple>
#ifdef BUILD_PTLAUNCH
#include <torch/extension.h>
#endif

#include "common/reduction.h"

template<typename scalar_t, ReductionType REDUCE>
int32_t segment_coo_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index,
                           const std::vector<int32_t> &index_size, std::tuple<scalar_t*, int32_t*> out, 
                           const std::vector<int32_t> &out_size, cudaStream_t stream);

template<typename scalar_t, ReductionType REDUCE>
int32_t segment_coo_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index,
                           const std::vector<int32_t> &index_size, const scalar_t *base, 
                           std::tuple<scalar_t*, int32_t*> out, const std::vector<int32_t> &out_size, 
                           cudaStream_t stream);

#ifdef BUILD_PTLAUNCH
//! \warning This function does not do any input legitimacy checking and is mainly used as a wrapper for testing the 
//! segment_coo_launch function on python.
int64_t segment_coo_ptlaunth(const torch::Tensor src, const torch::Tensor index, 
                             const torch::optional<torch::Tensor> base, const std::string& reduce, torch::Tensor out, 
                             torch::optional<torch::Tensor> arg_out);
#endif

#endif // TRTS_SEGMENT_COO_H