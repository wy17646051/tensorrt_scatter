import tempfile

import numpy as np
import pytest
import tensorrt as trt
import torch
import torch_scatter
from torch.onnx import register_custom_op_symbolic

import testing
from example.script import symbolic
from example.script.model import SegmentCOOExample

for _reduce in ['_sum', '_add', '_mean', '_min', '_max', '']:
    register_custom_op_symbolic(f'torch_scatter::segment{_reduce}_coo', getattr(symbolic, f'segment{_reduce}_coo'), 11)

cases_general = [
    (
        [1, 2, 3, 4, 5, 6],  # src
        [0, 0, 1, 1, 1, 3],  # index
        [-2, -2, -2, -2],    # base
        4,                   # dim_size
    ),
    (
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],  # src
        [0, 0, 1, 1, 1, 3],                                   # index
        [[-2, -2], [-2, -2], [-2, -2], [-2, -2]],             # base
        4,                                                    # dim_size
    ),
    (
        [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]],  # src
        [[0, 0, 1, 1, 1, 3], [0, 0, 0, 1, 1, 2]],     # index
        [[-2, -2, -2, -2], [-2, -2, -2, -2]],         # base
        4,                                            # dim_size
    ),
    (
        [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],          # src
        [[0, 0, 1], [0, 2, 2]],                                            # index
        [[[-2, -2], [-2, -2], [-2, -2]], [[-2, -2], [-2, -2], [-2, -2]]],  # base
        3,                                                                 # dim_size
    ),
    (
        [[1, 3], [2, 4]],  # src
        [[0, 0], [0, 0]],  # index
        [[-2], [-2]],      # base
        1,                 # dim_size
    ),
    (
        [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],  # src
        [[0, 0], [0, 0]],                      # index
        [[[-2, -2]], [[-2, -2]]],              # base
        1,                                     # dim_size
    )
]
cases_reduce = ['sum', 'add', 'mean', 'min', 'max']
cases_with_base = [True, False]
cases_with_arg_out = [True, False]
cases_dtype = [torch.float32, torch.half]


@pytest.mark.parametrize(
    'src,index,base,dim_size,reduce,with_base,with_arg_out,dtype',
    testing.case_composite(cases_general, cases_reduce, cases_with_base, cases_with_arg_out, cases_dtype)
)
def test_segment_coo_launth(src, index, base, dim_size, reduce, with_base, with_arg_out, dtype):
    src = torch.tensor(src, dtype=dtype).cuda()
    index = torch.tensor(index, dtype=torch.int64).cuda()
    base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
    
    except_out_tuple = getattr(torch_scatter, f'segment_{reduce}_coo')(
        src, index, base.clone() if base is not None else base, dim_size)
    if reduce not in ['min', 'max']:
        except_out_tuple = (except_out_tuple, None)
    except_out, except_arg_out = except_out_tuple
    
    # The `expand` operation in pytorch_scatter is implemented by converting the ONNX `Expand` operator, and the 
    # segment_coo_lauch function does not implement related logic.
    # TensorRT Plugin or ONNX does not need to consider `contiguous` operations, so no relevant logic is designed in 
    # the scatter_lauch function.
    index = index.expand(src.shape[:index.dim()]).contiguous()
    # Reduce type `sum` and `add` are the same, so only implement `sum` in the `scatter_launth` function.
    reduce = 'sum' if reduce == 'add' else reduce

    out = src.new_empty([s if i != index.dim()-1 else dim_size for i, s in enumerate(src.shape)])
    # In order to improve the calculation speed, even if the reduce is `max` or `min`, we are allowed to not calculate 
    # `arg_out`, but `out` must be calculated, because the former will depend on the latter.
    arg_out = index.new_empty(out.shape) if with_arg_out else None
    torch.ops.tensorrt_scatter.segment_coo_ptlaunth(src, index, base, reduce, out, arg_out)

    assert torch.allclose(out, except_out) and \
           (not with_arg_out or reduce not in ['min', 'max'] or torch.allclose(arg_out, except_arg_out))


@pytest.mark.parametrize(
    'src,index,base,dim_size,reduce,with_base,with_arg_out,dtype',
    testing.case_composite(cases_general, cases_reduce, cases_with_base, cases_with_arg_out, cases_dtype)
)
def test_segment_coo_plugin(src, index, base, dim_size, reduce, with_base, with_arg_out, dtype):
    with tempfile.TemporaryDirectory() as temp_dir:
        src = torch.tensor(src, dtype=dtype).cuda()
        index = torch.tensor(index, dtype=torch.int64).cuda()
        base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
        model = SegmentCOOExample(dim_size, reduce).cuda()

        with torch.no_grad():
            except_out_tuple = model(src, index, base.clone() if with_base else None)
            if reduce not in ['min', 'max']:
                except_out_tuple = (except_out_tuple.cpu().numpy(), None)
            else:
                except_out_tuple = tuple(t.cpu().numpy() for t in except_out_tuple)
            except_out, except_arg_out = except_out_tuple
        
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
        dim = index.dim() - 1        
        out_shape = [s if i != dim else dim_size for i, s in enumerate(src.shape)]
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
