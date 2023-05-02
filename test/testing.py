import ctypes
import os.path as osp
import sys
from itertools import product
from typing import Dict, List

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import torch

CURRENT_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_DIR = osp.dirname(CURRENT_DIR)
LIBTENSORRTSCATTER = osp.join(PROJECT_DIR, 'build', 'libtensorrtscatter.so')

ctypes.CDLL(LIBTENSORRTSCATTER)
torch.ops.load_library(LIBTENSORRTSCATTER)
sys.path.insert(1, PROJECT_DIR)
import example


def case_composite(*cases: List[Dict]) -> List:
    cases_composition = []
    for _cases in product(*cases):
        case = []
        for _case in _cases:
            if isinstance(_case, tuple):
                case.extend(_case)
            else:
                case.append(_case)        
        cases_composition.append(tuple(case))
    return cases_composition


def export_test_onnx(export_dir, model, args, input_names, output_names, opset_version=11, **kwargs):
    is_training = model.training
    model.eval()

    torch.onnx.export(
        model = model,
        args = args,
        f = osp.join(export_dir, 'tensorrt_scatter_test.onnx'),
        input_names = input_names,
        output_names = output_names,
        opset_version = opset_version,
        **kwargs
    )

    if is_training:
        model.train()


def export_test_trt(export_dir, trt_logger):
    trt.init_libnvinfer_plugins(trt_logger, '')

    builder = trt.Builder(trt_logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network_parser = trt.OnnxParser(network, trt_logger)
    success = network_parser.parse_from_file(osp.join(export_dir, 'tensorrt_scatter_test.onnx'))
    assert success, 'Failed to parse the ONNX file.'
    
    profile = builder.create_optimization_profile()
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 MiB
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    with open(osp.join(export_dir, 'tensorrt_scatter_test.engine'), 'wb') as f:
        f.write(serialized_engine)


def execute_test_trt(export_dir, trt_logger, inputs, outputs):
    trt.init_libnvinfer_plugins(trt_logger, '')
    with open(osp.join(export_dir, 'tensorrt_scatter_test.engine'), 'rb') as f:
        engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())

    outputs_shape = [_out.shape for _out in outputs]
    outputs_dtype = [_out.dtype for _out in outputs]
    inputs = [cuda.to_device(_in) for _in in inputs]
    outputs = [cuda.to_device(_out) for _out in outputs]

    context = engine.create_execution_context()
    context.execute_v2(inputs+outputs)

    final_outputs = []
    for _out, _shape, _dtype in zip(outputs, outputs_shape, outputs_dtype):
        final_outputs.append(cuda.from_device(_out, _shape, _dtype))
    
    return tuple(final_outputs)