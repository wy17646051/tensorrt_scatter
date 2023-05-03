import tempfile

import numpy as np
import pytest
import tensorrt as trt
import torch
import torch_scatter
from torch.onnx import register_custom_op_symbolic

import testing
from example.script import symbolic
from example.script.model import GatherCSRExample

register_custom_op_symbolic(f'torch_scatter::gather_csr', symbolic.gather_csr, 11)


cases_general = [
    (
        [1, 2, 3, 4],             #  src
        [0, 2, 5, 5, 6],          #  indptr
        [-2, -2, -2, -2, -2, -2], #  base
    ),
    (
        [[1, 2], [3, 4], [5, 6], [7, 8]],                            #  src
        [0, 2, 5, 5, 6],                                             #  indptr
        [[-2, -2], [-2, -2], [-2, -2], [-2, -2], [-2, -2], [-2, -2]] #  base
    ),
    (
        [[1, 3, 5, 7], [2, 4, 6, 8]],                         #  src
        [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],                   #  indptr
        [[-2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2]], #  base
    ),
    (
        [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],         #  src
        [[0, 2, 3, 3], [0, 1, 1, 3]],                                     #  indptr
        [[[-2, -2], [-2, -2], [-2, -2]], [[-2, -2], [-2, -2], [-2, -2]]], #  base
    ),
    (
        [[1], [2]],           #  src
        [[0, 2], [0, 2]],     #  indptr
        [[-2, -2], [-2, -2]], #  base
    ),
    (
        [[[1, 1]], [[2, 2]]],                         #  src
        [[0, 2], [0, 2]],                             #  indptr
        [[[-2, -2], [-2, -2]], [[-2, -2], [-2, -2]]], #  base
    )
]
cases_with_base = [True, False]
cases_dtype = [torch.float32, torch.half]


@pytest.mark.parametrize(
    'src,indptr,base,with_base,dtype', 
    testing.case_composite(cases_general,cases_with_base, cases_dtype)
)
def test_gather_csr_launth(src, indptr, base, with_base, dtype):
    src = torch.tensor(src, dtype=dtype).cuda()
    indptr = torch.tensor(indptr, dtype=torch.int64).cuda()
    base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
    
    except_out = torch_scatter.gather_csr(src, indptr, base.clone() if base is not None else base)

    # The `expand` operation in pytorch_scatter is implemented by converting the ONNX `Expand` operator, and the 
    # gather_csr_lauch function does not implement related logic.
    # TensorRT Plugin or ONNX does not need to consider `contiguous` operations, so no relevant logic is designed in 
    # the scatter_lauch function.
    indptr = indptr.expand(*src.shape[:indptr.dim()-1], indptr.shape[-1]).contiguous()

    out = src.new_empty([s if i != indptr.dim()-1 else max(indptr.flatten()[-1], 0) for i, s in enumerate(src.shape)])
    torch.ops.tensorrt_scatter.gather_csr_ptlaunth(src, indptr, base, out)

    assert torch.allclose(out, except_out)


@pytest.mark.parametrize(
    'src,indptr,base,dtype', 
    testing.case_composite(cases_general, cases_dtype)
)
def test_gather_csr_plugin(src, indptr, base, dtype):
    with tempfile.TemporaryDirectory() as temp_dir:
        src = torch.tensor(src, dtype=dtype).cuda()
        indptr = torch.tensor(indptr, dtype=torch.int64).cuda()
        base = torch.tensor(base, dtype=dtype).cuda()
        model = GatherCSRExample().cuda()

        with torch.no_grad():
            except_out = model(src, indptr, base)
            except_out = except_out.cpu().numpy()
        
        testing.export_test_onnx(temp_dir, model, 
                                 (src, indptr, base),
                                 ['src', 'indptr', 'base'],
                                 ['out'])

        trt_logger = trt.Logger(trt.Logger.WARNING)
        testing.export_test_trt(temp_dir, trt_logger)        

        inputs = [src.cpu().numpy(), indptr.cpu().numpy().astype(np.int32), base.cpu().numpy()]
        outputs = [np.empty(base.shape, dtype=inputs[0].dtype)]

        out = testing.execute_test_trt(temp_dir, trt_logger, inputs, outputs)[0]

    assert np.allclose(out, except_out)
