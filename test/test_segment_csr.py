import tempfile

import numpy as np
import pytest
import tensorrt as trt
import torch
import torch_scatter

import testing
from example.script.model import SegmentCSRExample

cases_general = [
    (
        [1, 2, 3, 4, 5, 6],  # src
        [0, 2, 5, 5, 6],     # indptr
        [-2, -2, -2, -2],    # base
    ),
    (
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],  # src
        [0, 2, 5, 5, 6],                                      # indptr
        [[-2, -2], [-2, -2], [-2, -2], [-2, -2]],             # base
    ),
    (
        [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]],  # src
        [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],           # indptr
        [[-2, -2, -2, -2], [-2, -2, -2, -2]],         # base
    ),
    (
        [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],          # src
        [[0, 2, 3, 3], [0, 1, 1, 3]],                                      # indptr
        [[[-2, -2], [-2, -2], [-2, -2]], [[-2, -2], [-2, -2], [-2, -2]]],  # base
    ),
    (
        [[1, 3], [2, 4]],  # src
        [[0, 2], [0, 2]],  # indptr
        [[-2], [-2]],      # base
    ),
    (
        [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],  # src
        [[0, 2], [0, 2]],                      # indptr
        [[[-2, -2]], [[-2, -2]]],              # base
    )
]
cases_reduce = ['sum', 'add', 'mean', 'min', 'max']
cases_with_base = [True, False]
cases_with_arg_out = [True, False]
cases_dtype = [torch.float32, torch.half]


@pytest.mark.parametrize(
    'src,indptr,base,reduce,with_base,with_arg_out,dtype',
    testing.case_composite(cases_general, cases_reduce, cases_with_base, cases_with_arg_out, cases_dtype)
)
def test_segment_csr_launth(src, indptr, base, reduce, with_base, with_arg_out, dtype):
    src = torch.tensor(src, dtype=dtype).cuda()
    indptr = torch.tensor(indptr, dtype=torch.int64).cuda()
    base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
    
    except_out_tuple = getattr(torch_scatter, f'segment_{reduce}_csr')(
        src, indptr, base.clone() if base is not None else base)
    if reduce not in ['min', 'max']:
        except_out_tuple = (except_out_tuple, None)
    except_out, except_arg_out = except_out_tuple
    
    # The `expand` operation in pytorch_scatter is implemented by converting the ONNX `Expand` operator, and the 
    # segment_csr_lauch function does not implement related logic.
    # TensorRT Plugin or ONNX does not need to consider `contiguous` operations, so no relevant logic is designed in 
    # the scatter_lauch function.
    indptr = indptr.expand(*src.shape[:indptr.dim()-1], indptr.shape[-1]).contiguous()
    # Reduce type `sum` and `add` are the same, so only implement `sum` in the `scatter_launth` function.
    reduce = 'sum' if reduce == 'add' else reduce

    out = src.new_empty([s if i != indptr.dim()-1 else max(indptr.shape[-1] - 1, 0) for i, s in enumerate(src.shape)])
    # In order to improve the calculation speed, even if the reduce is `max` or `min`, we are allowed to not calculate 
    # `arg_out`, but `out` must be calculated, because the former will depend on the latter.
    arg_out = indptr.new_empty(out.shape) if with_arg_out else None
    
    torch.ops.tensorrt_scatter.segment_csr_ptlaunth(src, indptr, base, reduce, out, arg_out)
    # print(f'out: {out}', out.shape)
    # print(f'except_out: {except_out}', except_out.shape)

    assert torch.allclose(out, except_out) and \
           (not with_arg_out or reduce not in ['min', 'max'] or torch.allclose(arg_out, except_arg_out))


@pytest.mark.parametrize(
    'src,indptr,base,reduce,with_base,with_arg_out,dtype',
    testing.case_composite(cases_general, cases_reduce, cases_with_base, cases_with_arg_out, cases_dtype)
)
def test_segment_csr_plugin(src, indptr, base, reduce, with_base, with_arg_out, dtype):
    with tempfile.TemporaryDirectory() as temp_dir:
        src = torch.tensor(src, dtype=dtype).cuda()
        indptr = torch.tensor(indptr, dtype=torch.int64).cuda()
        base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
        model = SegmentCSRExample(reduce).cuda()

        with torch.no_grad():
            except_out_tuple = model(src, indptr, base.clone() if with_base else None)
            if reduce not in ['min', 'max']:
                except_out_tuple = (except_out_tuple.cpu().numpy(), None)
            else:
                except_out_tuple = tuple(t.cpu().numpy() for t in except_out_tuple)
            except_out, except_arg_out = except_out_tuple
        
        _base = base.clone() if with_base else None
        testing.export_test_onnx(temp_dir, model, 
                                 (src, indptr, _base) if with_base else (src, indptr),
                                 ['src', 'indptr', 'base'] if with_base else ['src', 'indptr'],
                                 ['out', 'arg_out'] if reduce in ['max', 'min'] else ['out'])

        trt_logger = trt.Logger(trt.Logger.WARNING)
        testing.export_test_trt(temp_dir, trt_logger)        

        inputs = [src.cpu().numpy(), indptr.cpu().numpy().astype(np.int32)]
        if with_base:
            inputs.append(base.cpu().numpy())
        dim = indptr.dim() - 1        
        out_shape = [s if i != dim else max(inputs[1].shape[-1] - 1, 0) for i, s in enumerate(src.shape)]
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