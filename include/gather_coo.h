#ifndef TRTS_GATHER_COO_H
#define TRTS_GATHER_COO_H

#include <vector>
#ifdef BUILD_PTLAUNCH
#include <torch/extension.h>
#endif

template<typename scalar_t>
int32_t gather_coo_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index,
                          const std::vector<int32_t> &index_size, scalar_t* out, cudaStream_t stream);

template<typename scalar_t>
int32_t gather_coo_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *index,
                          const std::vector<int32_t> &index_size, const scalar_t *base, scalar_t* out, 
                          cudaStream_t stream);

#ifdef BUILD_PTLAUNCH
//! \warning This function does not do any input legitimacy checking and is mainly used as a wrapper for testing the 
//! gather_coo_launch function on python.
int64_t gather_coo_ptlaunth(const torch::Tensor src, const torch::Tensor index, 
                            const torch::optional<torch::Tensor> base, torch::Tensor out);
#endif

#endif // TRTS_GATHER_COO_H