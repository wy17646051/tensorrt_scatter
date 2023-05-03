import tempfile

import numpy as np
import pytest
import tensorrt as trt
import torch
import torch_scatter

import testing
from example.script.model import GatherCOOExample

cases_general = [
    (
        [1, 2, 3, 4],             #  src
        [0, 0, 1, 1, 1, 3],       #  index
        [-2, -2, -2, -2, -2, -2], #  base
    ),
    (
        [[1, 2], [3, 4], [5, 6], [7, 8]],                            #  src
        [0, 0, 1, 1, 1, 3],                                          #  index
        [[-2, -2], [-2, -2], [-2, -2], [-2, -2], [-2, -2], [-2, -2]] #  base
    ),
    (
        [[1, 3, 5, 7], [2, 4, 6, 8]],                         #  src
        [[0, 0, 1, 1, 1, 3], [0, 0, 0, 1, 1, 2]],             #  index
        [[-2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2]], #  base
    ),
    (
        [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],         #  src
        [[0, 0, 1], [0, 2, 2]],                                           #  index
        [[[-2, -2], [-2, -2], [-2, -2]], [[-2, -2], [-2, -2], [-2, -2]]], #  base
    ),
    (
        [[1], [2]],           #  src
        [[0, 0], [0, 0]],     #  index
        [[-2, -2], [-2, -2]], #  base
    ),
    (
        [[[1, 1]], [[2, 2]]],                         #  src
        [[0, 0], [0, 0]],                             #  index
        [[[-2, -2], [-2, -2]], [[-2, -2], [-2, -2]]], #  base
    )
]
cases_with_base = [True, False]
cases_dtype = [torch.float32, torch.half]


@pytest.mark.parametrize(
    'src,index,base,with_base,dtype', 
    testing.case_composite(cases_general,cases_with_base, cases_dtype)
)
def test_segment_coo_launth(src, index, base, with_base, dtype):
    src = torch.tensor(src, dtype=dtype).cuda()
    index = torch.tensor(index, dtype=torch.int64).cuda()
    base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
    
    except_out = torch_scatter.gather_coo(src, index, base.clone() if base is not None else base)

    # The `expand` operation in pytorch_scatter is implemented by converting the ONNX `Expand` operator, and the 
    # gather_coo_lauch function does not implement related logic.
    # TensorRT Plugin or ONNX does not need to consider `contiguous` operations, so no relevant logic is designed in 
    # the scatter_lauch function.
    index = index.expand(*src.shape[:index.dim()-1], index.shape[-1]).contiguous()

    out = src.new_empty([s if i != index.dim()-1 else index.shape[-1] for i, s in enumerate(src.shape)])
    torch.ops.tensorrt_scatter.gather_coo_ptlaunth(src, index, base, out)

    assert torch.allclose(out, except_out)


@pytest.mark.parametrize(
    'src,index,base,with_base,dtype', 
    testing.case_composite(cases_general,cases_with_base, cases_dtype)
)
def test_gather_coo_plugin(src, index, base, with_base, dtype):
    with tempfile.TemporaryDirectory() as temp_dir:
        src = torch.tensor(src, dtype=dtype).cuda()
        index = torch.tensor(index, dtype=torch.int64).cuda()
        base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
        model = GatherCOOExample().cuda()

        with torch.no_grad():
            except_out = model(src, index, base.clone() if with_base else None)
            except_out = except_out.cpu().numpy()
        
        _base = base.clone() if with_base else None
        testing.export_test_onnx(temp_dir, model, 
                                 (src, index, _base) if with_base else (src, index),
                                 ['src', 'index', 'base'] if with_base else ['src', 'index'],
                                 ['out'])

        trt_logger = trt.Logger(trt.Logger.WARNING)
        testing.export_test_trt(temp_dir, trt_logger)        

        inputs = [src.cpu().numpy(), index.cpu().numpy().astype(np.int32)]
        if with_base:
            inputs.append(base.cpu().numpy())
        dim = index.dim() - 1        
        out_shape = [s if i != dim else index.shape[dim] for i, s in enumerate(src.shape)]
        outputs = [np.empty(out_shape, dtype=inputs[0].dtype)]

        out = testing.execute_test_trt(temp_dir, trt_logger, inputs, outputs)[0]

    assert np.allclose(out, except_out)
