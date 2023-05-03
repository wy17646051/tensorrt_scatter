# TensorRT Scatter

**[TensorRT](https://developer.nvidia.com/tensorrt) Plugin** of corresponding **[PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter/tree/master) operators**.

<p align="center">
  <img width="76%" src="https://github.com/wy17646051/tensorrt_scatter/tree/master/docs/logo.png" />
</p>

---

At present, the project is only tested on **TensorRT 8.5.x** and **CUDA 11.6**, *this does not mean that other versions cannot run, but it should be used with caution*.

| Supporting Operators                                         | TensorRT Version | CUDA Version |
| ------------------------------------------------------------ | ---------------- | ------------ |
| [**scatter**](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html) (sum, add, mean, mul, min, max) | 8.5.x            | 11.6         |
| [**segment_coo**](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html) (sum, add, mean, min, max) | 8.5.x            | 11.6         |
| **gather_coo**                                               | 8.5.x            | 11.6         |
| [**segment_csr**](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html) (sum, add, mean, min, max) | 8.5.x            | 11.6         |
| **gather_csr**                                               | 8.5.x            | 11.6         |

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
from example.script.symbolic import register_symbolic
register_symbolic(op_name=None, opset_version=9)
```

## Example

The Project produce some simple **example models** based on PyTorch and provided some test data. The original form of the test data is a 3D point cloud of shape *[N, 5]* (3D coordinates in the first three dimensions and point attributes in the last two dimensions). The model and data loading logic are implemented in `example/script/model.py`.

In addition, we provide **pytorch -> onnx** and **onnx -> tenerrt** transformation scripts based on the example model in `example/script/export.py`, which can be run as follows:

```shell
 export PYTHONPATH="example"
 python example/script/export.py --model scatter_example --trt --onnx
```