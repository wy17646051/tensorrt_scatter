import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

import tensorrt as trt
import ctypes
import tempfile
import os.path as osp
import pytest
import testing
import torch
import torch_scatter
from torch.onnx import (OperatorExportTypes, TrainingMode,
                        register_custom_op_symbolic)
from torch_scatter.utils import broadcast

from example.script import symbolic
from example.script.model import ScatterExample

for _reduce in ['_sum', '_add', '_mul', '_mean', '_min', '_max', '']:
    register_custom_op_symbolic(f'torch_scatter::scatter{_reduce}', getattr(symbolic, f'scatter{_reduce}'), 11)


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
    
    except_out_tuple = getattr(torch_scatter, 'scatter_' + reduce)(
        src, index, dim, base.clone() if base is not None else base, dim_size)
    if reduce not in ['min', 'max']:
        except_out_tuple = (except_out_tuple, None)
    except_out, except_arg_out = except_out_tuple

    index = broadcast(index, src, dim)
    # Reduce type `sum` and `add` are the same, so only implement `sum` in the `scatter_launth` function.
    reduce = 'sum' if reduce == 'add' else reduce

    _dim = dim if dim >= 0 else src.dim() + dim
    out = src.new_empty([s if i != _dim else dim_size for i, s in enumerate(src.shape)])
    # In order to improve the calculation speed, even if the reduce is `max` or `min`, we are allowed to not calculate 
    # `arg_out`, but `out` must be calculated, because the former will depend on the latter.
    arg_out = index.new_empty(out.shape) if with_arg_out else None
    # The `broadcast` operation in pytorch_scatter is implemented by converting the ONNX `Expand` operator, and the 
    # scatter_lauch function does not implement related logic.
    # TensorRT Plugin or ONNX does not need to consider `contiguous`` operations, so no relevant logic is designed in 
    # the scatter_lauch function.
    torch.ops.tensorrt_scatter.scatter_ptlaunth(src, index, dim, base, reduce, out, arg_out)

    assert torch.allclose(out, except_out) and \
           (not with_arg_out or reduce not in ['min', 'max'] or torch.allclose(arg_out, except_arg_out))


def _export_onnx(model, src, index, base, reduce, with_base, temp_dir):
    model.eval()
    torch.onnx.export(
        model = model,
        args = (src, index, base) if with_base else (src, index),
        f = osp.join(temp_dir, 'model.onnx'),
        export_params = True,
        verbose = False,
        training = TrainingMode.EVAL,
        input_names = ['src', 'index', 'base'] if with_base else ['src', 'index'],
        output_names = ['out', 'arg_out'] if reduce in ['max', 'min'] else ['out'],
        operator_export_type = OperatorExportTypes.ONNX,
        opset_version = 11,
        do_constant_folding = True,
        keep_initializers_as_inputs = None,
        custom_opsets = None,
        export_modules_as_functions = False
    )


def _export_trt(trt_logger, temp_dir):
    builder = trt.Builder(trt_logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network_parser = trt.OnnxParser(network, trt_logger)
    success = network_parser.parse_from_file(osp.join(temp_dir, 'model.onnx'))
    assert success, 'Failed to parse the ONNX file.'
    
    profile = builder.create_optimization_profile()
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 MiB
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    with open(osp.join(temp_dir, 'model.engine'), 'wb') as f:
        f.write(serialized_engine)


@pytest.mark.parametrize(
    'src,index,dim,base,dim_size,reduce,with_base,with_arg_out,dtype',
    testing.case_composite(cases_general, cases_reduce, cases_with_base, cases_with_arg_out, cases_dtype)
)
def test_scatter_plugin(src, index, dim, base, dim_size, reduce, with_base, with_arg_out, dtype):
    # Note that we do not test `with_arg_out` here, because to keep with the same parameters as operators in 
    # `pytorch_scatter`, the plugin adopts the same behavior as the operators in pytorch-scatter.
    with tempfile.TemporaryDirectory() as temp_dir:
        src = torch.tensor(src, dtype=dtype).cuda()
        index = torch.tensor(index, dtype=torch.int64).cuda()
        base = torch.tensor(base, dtype=dtype).cuda() if with_base else None
        model = ScatterExample(dim, dim_size, reduce).cuda()

        with torch.no_grad():
            except_out_tuple = model(src, index, base.clone() if base is not None else base)
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
        _export_onnx(model, src, index, base.clone() if base is not None else base, reduce, with_base, temp_dir)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, '')
        _export_trt(trt_logger, temp_dir)

        src, index = src.cpu().numpy(), index.cpu().numpy().astype(np.int32)
        src_shape, src_dtype, index_dtype = src.shape, src.dtype, index.dtype
        src, index = cuda.to_device(src), cuda.to_device(index)
        base = cuda.to_device(base.cpu().numpy()) if base is not None else None     
        
        _dim = dim if dim >= 0 else len(src_shape) + dim
        out_shape = [s if i != _dim else dim_size for i, s in enumerate(src_shape)]
        out = cuda.to_device(np.empty(out_shape, dtype=src_dtype))
        arg_out = cuda.to_device(np.empty(out_shape, dtype=index_dtype)) if reduce in ['min', 'max'] else None

        buffer = [src, index]
        if with_base:
            buffer.append(base)
        buffer.append(out)
        if reduce in ['min', 'max']:
            buffer.append(arg_out)
        
        with open(osp.join(temp_dir, 'model.engine'), "rb") as f:
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        context.execute_v2(buffer)

        out = cuda.from_device(out, out_shape, src_dtype)
        arg_out = cuda.from_device(arg_out, out_shape, index_dtype) if reduce in ['min', 'max'] else None

    assert np.allclose(out, except_out) and \
           (reduce not in ['min', 'max'] or np.allclose(arg_out, except_arg_out))
