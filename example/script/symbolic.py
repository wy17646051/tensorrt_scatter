import warnings

from torch.onnx import symbolic_helper


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