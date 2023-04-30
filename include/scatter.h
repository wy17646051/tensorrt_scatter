#ifndef TRTS_SCATTER_H
#define TRTS_SCATTER_H

#include <vector>
#include <tuple>
#ifdef BUILD_PTLAUNCH
#include <torch/extension.h>
#endif

#include "common/reduction.h"

//!
//! \brief Entry function of scatter operator.
//!
//! \param src The source tensor.
//! \param src_size The size of source tensor.
//! \param index  The indices of elements to scatter.
//! \param dim The axis along which to index.
//! \param out A Tuple of destination and indices tensor. (indices will be nullptr if REDUCE is not "MAX" or "MIN")
//! \param out_size The size of destination tensor.
//! \param stream The cuda stream.
//!
//! \return 0 for success, else non-zero (which will cause engine termination).
//!
template <typename scalar_t, ReductionType REDUCE>
int32_t scatter_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index, int32_t dim,
                       std::tuple<scalar_t*, int32_t*> out, const std::vector<int32_t> &out_size, cudaStream_t stream);

//!
//! \brief Entry function of scatter operator. (Supports out initialization by "base")
//!
//! \param src The source tensor.
//! \param src_size The size of source tensor.
//! \param index  The indices of elements to scatter.
//! \param dim The axis along which to index.
//! \param base The base tensor to initialize destination tensor.
//! \param out A Tuple of destination and indices tensor. (indices will be nullptr if REDUCE is not "MAX" or "MIN")
//! \param out_size The size of destination tensor.
//! \param stream The cuda stream.
//!
//! \return 0 for success, else non-zero (which will cause engine termination).
//!
template <typename scalar_t, ReductionType REDUCE>
int32_t scatter_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index, int32_t dim,
                       const scalar_t *base, std::tuple<scalar_t*, int32_t*> out, const std::vector<int32_t> &out_size,
                       cudaStream_t stream);

#ifdef BUILD_PTLAUNCH
//! \warning This function does not do any input legitimacy checking and is mainly used as a wrapper for testing the 
//! scatter_launch function on python.
int64_t scatter_ptlaunth(const torch::Tensor src, const torch::Tensor index, int64_t dim, 
                         const torch::optional<torch::Tensor> base, const std::string& reduce, torch::Tensor out, 
                         torch::optional<torch::Tensor> arg_out);
#endif

#endif // TRTS_SCATTER_H