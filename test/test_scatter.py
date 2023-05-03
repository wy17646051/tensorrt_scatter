import tempfile

import numpy as np
import pytest
import tensorrt as trt
import torch
import torch_scatter
from torch_scatter.utils import broadcast

import testing
from example.script.model import ScatterExample

cases_general = [
    (
        [1, 3, 2, 4, 5, 6],  # src
        [0, 1, 0, 1, 1, 3],  # index
        -1,                  # dim
        [-2, -2, -2, -2],    # base
        4,                   # dim_size 
    ),
    (
        [[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]],  # src
        [0, 1, 0, 1, 1, 3],                                  # index
        0,                                                   # dim
        [[-2, -2], [-2, -2], [-2, -2], [-2, -2]],            # base
        4,                                                   # dim_size
    ),
    (
        [[1, 5, 3, 7, 9, 11], [2, 4, 8, 6, 10, 12]],  # src
        [0, 1, 0, 1, 1, 3],                           # index
        1,                                            # dim
        [[-2, -2, -2, -2], [-2, -2, -2, -2]],         # base
        4,                                            # dim_size
    ),
    (
        [[[1, 2], [5, 6], [3, 4]], [[10, 11], [7, 9], [12, 13]]],          # src
        [[0, 1, 0], [2, 0, 2]],                                            # index
        1,                                                                 # dim
        [[[-2, -2], [-2, -2], [-2, -2]], [[-2, -2], [-2, -2], [-2, -2]]],  # base
        3,                                                                 # dim_size
    ),
    (
        [[1, 3], [2, 4]],  # src
        [[0, 0], [0, 0]],  # index
        1,                 # dim
        [[-2], [-2]],      # base
        1,                 # dim_size
    ),
    (
        [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],  # src
        [[0, 0], [0, 0]],                      # index
        1,                                     # dim
        [[[-2, -2]], [[-2, -2]]],              # base
        1,                                     # dim_size
    )
]
cases_reduce = ['sum', 'add', 'mean', 'min', 'max', 'mul']
cases_with_base = [True, False]
cases_with_arg_out = [True, False]
cases_dtype = [torch.float32, torch.half]


@pytest.mark.parametrize(
    'src,index,dim,base,dim_size,reduce,with_base,with_arg_out,dtype',
    testing.case_composite(cases_general, cases_reduce, cases_with_base, cases_with_arg_out, cases_dtype)
)
def test_scatter_launth(src, index, dim, base, dim_size, reduce, with_base, with_arg_out, dtype):
    src = torch.tensor(src, dtype=dtype).cuda()
    index = torch.tensor(index, dtype=torch.int64).cuda()
    base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
    
    except_out_tuple = getattr(torch_scatter, f'scatter_{reduce}')(
        src, index, dim, base.clone() if base is not None else base, dim_size)
    if reduce not in ['min', 'max']:
        except_out_tuple = (except_out_tuple, None)
    except_out, except_arg_out = except_out_tuple
    
    # The `broadcast` operation in pytorch_scatter is implemented by converting the ONNX `Expand` operator, and the 
    # scatter_lauch function does not implement related logic.
    # TensorRT Plugin or ONNX does not need to consider `contiguous` operations, so no relevant logic is designed in 
    # the scatter_lauch function.
    index = broadcast(index, src, dim).contiguous()
    # Reduce type `sum` and `add` are the same, so only implement `sum` in the `scatter_launth` function.
    reduce = 'sum' if reduce == 'add' else reduce

    _dim = dim if dim >= 0 else src.dim() + dim
    out = src.new_empty([s if i != _dim else dim_size for i, s in enumerate(src.shape)])
    # In order to improve the calculation speed, even if the reduce is `max` or `min`, we are allowed to not calculate 
    # `arg_out`, but `out` must be calculated, because the former will depend on the latter.
    arg_out = index.new_empty(out.shape) if with_arg_out else None
    torch.ops.tensorrt_scatter.scatter_ptlaunth(src, index, dim, base, reduce, out, arg_out)

    assert torch.allclose(out, except_out) and \
           (not with_arg_out or reduce not in ['min', 'max'] or torch.allclose(arg_out, except_arg_out))


@pytest.mark.parametrize(
    'src,index,dim,base,dim_size,reduce,with_base,with_arg_out,dtype',
    testing.case_composite(cases_general, cases_reduce, cases_with_base, cases_with_arg_out, cases_dtype)
)
def test_scatter_plugin(src, index, dim, base, dim_size, reduce, with_base, with_arg_out, dtype):
    with tempfile.TemporaryDirectory() as temp_dir:
        src = torch.tensor(src, dtype=dtype).cuda()
        index = torch.tensor(index, dtype=torch.int64).cuda()
        base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
        model = ScatterExample(dim, dim_size, reduce).cuda()

        with torch.no_grad():
            except_out_tuple = model(src, index, base.clone() if with_base else None)
            if reduce not in ['min', 'max']:
                except_out_tuple = (except_out_tuple.cpu().numpy(), None)
            else:
                except_out_tuple = tuple(t.cpu().numpy() for t in except_out_tuple)
            except_out, except_arg_out = except_out_tuple
        
        # The torch_scatter `scatter_xxx` operators of type `sum`, `add` and `mean` are implemented using the native 
        # pytorch. In order for the `pytorch2onnx` converter to correctly recognize and apply our custom symbolic 
        # functions, we need to place these scatter operators for the corresponding cuda versions for placeholders.
        if reduce in ['sum', 'add']:
            model.scatter_op = torch.ops.torch_scatter.scatter_sum
        if reduce in ['mean']:
            model.scatter_op = torch.ops.torch_scatter.scatter_mean
        _base = base.clone() if with_base else None
        testing.export_test_onnx(temp_dir, model, 
                                 (src, index, _base) if with_base else (src, index),
                                 ['src', 'index', 'base'] if with_base else ['src', 'index'],
                                 ['out', 'arg_out'] if reduce in ['max', 'min'] else ['out'])

        trt_logger = trt.Logger(trt.Logger.WARNING)
        testing.export_test_trt(temp_dir, trt_logger)        

        inputs = [src.cpu().numpy(), index.cpu().numpy().astype(np.int32)]
        if with_base:
            inputs.append(base.cpu().numpy())
        _dim = dim if dim >= 0 else len(src.shape) + dim
        out_shape = [s if i != _dim else dim_size for i, s in enumerate(src.shape)]
        outputs = [np.empty(out_shape, dtype=inputs[0].dtype)]
        # Note that we do not test `with_arg_out` here, because to keep with the same parameters as operators in 
        # `pytorch_scatter`, the plugin adopts the same behavior as the operators in pytorch-scatter.
        if reduce in ['min', 'max']:
            outputs.append(np.empty(out_shape, dtype=inputs[1].dtype))

        out_tuple = testing.execute_test_trt(temp_dir, trt_logger, inputs, outputs)
        if reduce not in ['min', 'max']:
            out_tuple = (out_tuple[0], None)
        out, arg_out = out_tuple

    assert np.allclose(out, except_out) and \
           (reduce not in ['min', 'max'] or np.allclose(arg_out, except_arg_out))
