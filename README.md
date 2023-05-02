# TensorRT Scatter

**[TensorRT](https://developer.nvidia.com/tensorrt) Plugin** of corresponding **[PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter/tree/master) operators**.

<p align="center">
  <img width="50%" src="https://raw.githubusercontent.com/rusty1s/pytorch_scatter/master/docs/source/_figures/add.svg?sanitize=true" />
</p>



---

At present, the project is only tested on TensorRT 8.5x and CUDA11.6 (*this does not mean that other versions cannot run, but it should be used with caution*). TensorRT Plugins corresponding to other operators will come soon, and test on more environments.

| Supporting Operators                                         | TensorRT Version | CUDA Version |
| ------------------------------------------------------------ | ---------------- | ------------ |
| [**scatter**](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html) (sum, add, mean, mul, min, max) | 8.5.x            | 11.6         |
| [**segment_coo**](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html) (sum, add, mean, min, max) | 8.5.x            | 11.6         |
| **gather_coo**                                               |                  |              |

## Installation

Before installing the project, make sure you have configured your **CUDA** environment based on the support list above and downloaded **TensorRT**.

### TensorRT Plugin

Build the project based on **CMake** as follows:

```shell
mkdir build && cd build
cmake .. -DTENSORRT_PREFIX_PATH="/The/TensorRT/path/you/downloaded" && make
```

### PyTorch Symbolic

The project additionally provides the **symbolic function** corresponding to the pytorch_scatter operator required by pytorch to export to onnx in `example/script/symbolic.py`. 

Make sure to **register** the **symbol function** before calling `torch.onnx.export` to export the onnx model, *e.g.*:

```python
from torch.onnx import register_custom_op_symbolic
from example.script import symbolic

for _reduce in ['_sum', '_add', '_mul', '_mean', '_min', '_max', '']:
    register_custom_op_symbolic(
      f'torch_scatter::scatter{_reduce}', getattr(symbolic, f'scatter{_reduce}'), 9)

for _reduce in ['_sum', '_add', '_mean', '_min', '_max', '']:
    register_custom_op_symbolic(f'torch_scatter::segment{_reduce}_coo', getattr(symbolic, f'segment{_reduce}_coo'), 9)

register_custom_op_symbolic(f'torch_scatter::gather_coo', symbolic.gather_coo, 9)
```

## Example

The Project produce some simple **example models** based on PyTorch and provided some test data. The original form of the test data is a 3D point cloud of shape *[N, 5]* (3D coordinates in the first three dimensions and point attributes in the last two dimensions). The model and data loading logic are implemented in `example/script/model.py`.

In addition, we provide **pytorch -> onnx** and **onnx -> tenerrt** transformation scripts based on the example model in `example/script/export.py`, which can be run as follows:

```shell
 export PYTHONPATH="example"
 python example/script/export.py --model scatter_example --trt --onnx
```