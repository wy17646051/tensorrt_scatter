import warnings

import torch
from torch.onnx import symbolic_helper


def _broadcast(g, src, other, dim):
    _get_rank = symbolic_helper._get_tensor_rank

    if dim < 0:
        dim = _get_rank(other) + dim

    if _get_rank(src) == 1:
        for _ in range(0, dim):
            src = symbolic_helper._unsqueeze_helper(g, src, axes_i=[0])

    for _ in range(_get_rank(src), _get_rank(other)):
        src = symbolic_helper._unsqueeze_helper(g, src, axes_i=[_get_rank(src)])

    shape = g.op("Shape", other)
    return g.op("Expand", src, shape)


@symbolic_helper.parse_args('v', 'v', 'i', 'v', 'i', 's')
def scatter(g, src, index, dim, out, dim_size, reduce):
    if dim_size is None:
        warnings.warn(
            "Warning: If use of TensorRT, please indicate `dim_size`. " 
            "The output shape of the Tensorrt plugin should be determined by the input shape. "
            "Dynamic calculation by inputs data is not allowed."
        )
    if reduce not in ['sum', 'mean', 'mul', 'min', 'max']:
        raise ValueError("Only support `sum`, `mean`, `mul`, `min`, `max` reduce type")
    
    index = _broadcast(g, index, src, dim)
    
    args = [src, index]
    if not symbolic_helper._is_none(out):
        args.append(out)
    
    kwargs = {
        'dim_i':dim, 
        'reduce_s': reduce 
    }
    if dim_size is not None:
        kwargs['dim_size_i'] = dim_size

    outputs = 2 if reduce in ['min', 'max'] else 1

    return g.op("tensorrt_scatter::TRTS_Scatter", *args, **kwargs, outputs=outputs)


def scatter_sum(g, src, index, dim, out, dim_size):
    return scatter(g, src, index, dim, out, dim_size, 'sum')


def scatter_add(g, src, index, dim, out, dim_size):
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(g, src, index, dim, out, dim_size):
    return scatter(g, src, index, dim, out, dim_size, 'mul')


def scatter_mean(g, src, index, dim, out, dim_size):
    return scatter(g, src, index, dim, out, dim_size, 'mean')


def scatter_min(g, src, index, dim, out, dim_size):
    return scatter(g, src, index, dim, out, dim_size, 'min')


def scatter_max(g, src, index, dim, out, dim_size):
    return scatter(g, src, index, dim, out, dim_size, 'max')


@symbolic_helper.parse_args('v', 'v', 'v', 'i', 's')
def segment_coo(g, src, index, out, dim_size, reduce):
    if dim_size is None:
        warnings.warn(
            "Warning: If use of TensorRT, please indicate `dim_size`. " 
            "The output shape of the Tensorrt plugin should be determined by the input shape. "
            "Dynamic calculation by inputs data is not allowed."
        )
    if reduce not in ['sum', 'mean', 'min', 'max']:
        raise ValueError("Only support `sum`, `mean`, `min`, `max` reduce type")
    
    index_rank = symbolic_helper._get_tensor_rank(index)
    size = symbolic_helper._get_tensor_sizes(src)[:index_rank]
    
    size = g.op("Constant", value_t=torch.LongTensor(size))
    index = g.op("Expand", index, size)
    
    args = [src, index]
    if not symbolic_helper._is_none(out):
        args.append(out)
    
    kwargs = {
        'reduce_s': reduce 
    }
    if dim_size is not None:
        kwargs['dim_size_i'] = dim_size

    outputs = 2 if reduce in ['min', 'max'] else 1

    return g.op("tensorrt_scatter::TRTS_SegmentCOO", *args, **kwargs, outputs=outputs)


def segment_sum_coo(g, src, index, out, dim_size):
    return segment_coo(g, src, index, out, dim_size, 'sum')


def segment_add_coo(g, src, index, out, dim_size):
    return segment_sum_coo(g, src, index, out, dim_size)


def segment_mean_coo(g, src, index, out, dim_size):
    return segment_coo(g, src, index, out, dim_size, 'mean')


def segment_min_coo(g, src, index, out, dim_size):
    return segment_coo(g, src, index, out, dim_size, 'min')


def segment_max_coo(g, src, index, out, dim_size):
    return segment_coo(g, src, index, out, dim_size, 'max')


@symbolic_helper.parse_args('v', 'v', 'v')
def gather_coo(g, src, index, out):   
    index_rank = symbolic_helper._get_tensor_rank(index)
    size = symbolic_helper._get_tensor_sizes(src)[:index_rank]
    size[index_rank-1] = symbolic_helper._get_tensor_sizes(index)[-1]
    
    size = g.op("Constant", value_t=torch.LongTensor(size))
    index = g.op("Expand", index, size)
    
    args = [src, index]
    if not symbolic_helper._is_none(out):
        args.append(out)

    return g.op("tensorrt_scatter::TRTS_GatherCOO", *args)


@symbolic_helper.parse_args('v', 'v', 'v', 's')
def segment_csr(g, src, indptr, out, reduce):
    if reduce not in ['sum', 'mean', 'min', 'max']:
        raise ValueError("Only support `sum`, `mean`, `min`, `max` reduce type")
    
    indptr_rank = symbolic_helper._get_tensor_rank(indptr)
    size = symbolic_helper._get_tensor_sizes(src)[:indptr_rank]
    size[indptr_rank-1] = symbolic_helper._get_tensor_sizes(indptr)[-1]
    
    size = g.op("Constant", value_t=torch.LongTensor(size))
    indptr = g.op("Expand", indptr, size)
    
    args = [src, indptr]
    if not symbolic_helper._is_none(out):
        args.append(out)
    
    kwargs = {'reduce_s': reduce}

    outputs = 2 if reduce in ['min', 'max'] else 1

    return g.op("tensorrt_scatter::TRTS_SegmentCSR", *args, **kwargs, outputs=outputs)


def segment_sum_csr(g, src, indptr, out):
    return segment_csr(g, src, indptr, out, 'sum')


def segment_add_csr(g, src, indptr, out):
    return segment_sum_csr(g, src, indptr, out)


def segment_mean_csr(g, src, indptr, out):
    return segment_csr(g, src, indptr, out, 'mean')


def segment_min_csr(g, src, indptr, out):
    return segment_csr(g, src, indptr, out, 'min')


def segment_max_csr(g, src, indptr, out):
    return segment_csr(g, src, indptr, out, 'max')


@symbolic_helper.parse_args('v', 'v', 'v')
def gather_csr(g, src, indptr, out):
    if out is None:
        warnings.warn(
            "Warning: The output shape cannot be calculated by the shape of `scr` or `inddptr`. If this operator is "
            "applied to TensorRT, `out` must be specified and its shape should be guaranteed to be fixed, otherwise an "
            "unknown error will occur."
        )

    indptr_rank = symbolic_helper._get_tensor_rank(indptr)
    size = symbolic_helper._get_tensor_sizes(src)[:indptr_rank]
    size[indptr_rank-1] = symbolic_helper._get_tensor_sizes(indptr)[-1]

    size = g.op("Constant", value_t=torch.LongTensor(size))
    indptr = g.op("Expand", indptr, size)
    
    args = [src, indptr]
    if not symbolic_helper._is_none(out):
        args.append(out)

    return g.op("tensorrt_scatter::TRTS_GatherCSR", *args)
