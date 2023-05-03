#ifndef TRTS_GATHER_CSR_H
#define TRTS_GATHER_CSR_H

#include <vector>
#ifdef BUILD_PTLAUNCH
#include <torch/extension.h>
#endif

template<typename scalar_t>
int32_t gather_csr_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *indptr, 
                          const std::vector<int32_t> &indptr_size, const scalar_t *base, scalar_t* out, 
                          const std::vector<int32_t> &out_size, cudaStream_t stream);
template<typename scalar_t>
int32_t gather_csr_launch(const scalar_t *src, const std::vector<int32_t> &src_size, const int32_t *indptr, 
                          const std::vector<int32_t> &indptr_size, scalar_t* out, const std::vector<int32_t> &out_size, 
                          cudaStream_t stream);

#ifdef BUILD_PTLAUNCH
//! \warning This function does not do any input legitimacy checking and is mainly used as a wrapper for testing the 
//! gather_csr_launch function on python.
int64_t gather_csr_ptlaunth(const torch::Tensor src, const torch::Tensor indptr, 
                            const torch::optional<torch::Tensor> base, torch::Tensor out);
#endif

#endif // TRTS_GATHER_Csr_H