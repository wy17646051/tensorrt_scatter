import argparse
import ctypes
from pathlib import Path

import tensorrt as trt
import torch
import yaml
import script.symbolic as symbolic
from script.model import ScatterExample, iter_data, scatter_prep_fn
from torch.onnx import (OperatorExportTypes, TrainingMode,
                        register_custom_op_symbolic)

LIBTENSORRTSCATTER = Path(__file__).parent.parent.parent /'build' / 'libtensorrtscatter.so'
ctypes.CDLL(LIBTENSORRTSCATTER)

for _reduce in ['_sum', '_add', '_mul', '_mean', '_min', '_max', '']:
    register_custom_op_symbolic(f'torch_scatter::scatter{_reduce}', getattr(symbolic, f'scatter{_reduce}'), 11)


@torch.no_grad()
def scatter_example_onnx_export(config_dict):
    torch.manual_seed(1)
    
    model = ScatterExample(**config_dict['model']).cuda()
    src, index, base = iter_data(scatter_prep_fn, **config_dict['data']).__next__()
    src, index, base = src.cuda(), index.cuda(), base.cuda() if base is not None else None
    dynamic_axes={
            'src': {
                0: 'batch_size',
                1: 'src_size'
            },
            'index': {
                0: 'src_size',
            },
            'out': {
                0: 'batch_size',
                1: 'dim_size'
            }
        }
    if config_dict['data']['with_base']:
        dynamic_axes['base'] = {
            0: 'batch_size',
            1: 'dim_size'
        }
    if config_dict['model']['reduce'] in ['max', 'min']:
        dynamic_axes['arg_out'] = {
            0: 'batch_size',
            1: 'dim_size'
        }

    model.eval()
    torch.onnx.export(
        model = model,
        args = (src, index, base) if config_dict['data']['with_base'] else (src, index),
        f = Path(__file__).parent.parent / config_dict['exports']['onnx_path'],
        export_params = True,
        verbose = False,
        training = TrainingMode.EVAL,
        input_names = ['src', 'index', 'base'] if config_dict['data']['with_base'] else ['src', 'index'],
        output_names = ['out', 'arg_out'] if config_dict['model']['reduce'] in ['max', 'min'] else ['out'],
        operator_export_type = OperatorExportTypes.ONNX,
        opset_version = 11,
        do_constant_folding = True,
        dynamic_axes = dynamic_axes,
        keep_initializers_as_inputs = None,
        custom_opsets = None,
        export_modules_as_functions = False
    )


def _is_plugin_registried(name):
    is_registried = False 
    for plugin in trt.get_plugin_registry().plugin_creator_list:
        if plugin.name == name:
            is_registried = True
            break
    return is_registried


def scatter_example_trt_export(config_dict):
    plugin_name = 'TRTS_Scatter'
    onnx_file = Path(__file__).parent.parent / config_dict['exports']['onnx_path']
    
    trt_logger = trt.Logger(trt.Logger.WARNING)

    trt.init_libnvinfer_plugins(trt_logger, '')
    if not _is_plugin_registried(plugin_name):
        raise RuntimeError(f'{plugin_name} plugin is not exist.')

    builder = trt.Builder(trt_logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network_parser = trt.OnnxParser(network, trt_logger)
    success = network_parser.parse_from_file(str(onnx_file))
    if not success:
        for idx in range(network_parser.num_errors):
            print(network_parser.get_error(idx))
        return
    
    profile = builder.create_optimization_profile()
    profile.set_shape('src', (1, 10000, 4), (1, 35000, 4), (1, 50000, 4))
    profile.set_shape('index', (10000,), (35000,), (50000,))
    if config_dict['data']['with_base']:
        profile.set_shape('base', (1, 10000, 4), (1, 10000, 4), (1, 10000, 4))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 MiB
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)

    with open(Path(__file__).parent.parent / config_dict['exports']['trt_path'], 'wb') as f:
        f.write(serialized_engine)


if __name__ == '__main__':
    with open(Path(__file__).parent / 'config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    parse = argparse.ArgumentParser(description='Export example.')
    parse.add_argument('--model', help='Model name.')
    parse.add_argument('--onnx', action='store_true', help='Export onnx.')
    parse.add_argument('--trt', action='store_true', help='Export tensorrt.')
    args = parse.parse_args()

    if args.onnx:
        scatter_example_onnx_export(config_dict[args.model])
    if args.trt:
        scatter_example_trt_export(config_dict[args.model])
    