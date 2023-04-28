from pathlib import Path

import numpy as np
import torch
import torch_scatter
from torch_scatter.utils import broadcast

class BaseExample(torch.nn.Module):

    def __init__(self, before=None, after=None):
        super(BaseExample, self).__init__()
        self.before = eval(before)
        self.after = eval(after)

    def forward(self, src, *args, **kwargs):
        if self.before is not None:
            src = self.before(src)
        out = self.forward_op(src, *args, **kwargs)
        if self.after is not None:
            out = self.after(out)
        return out

    def forward_op(self, src, *args, **kwargs):
        raise NotImplementedError


class ScatterExample(BaseExample):

    def __init__(self, dim, dim_size, reduce, before=None, after=None):
        super(ScatterExample, self).__init__(before, after)
        self.dim = dim
        self.dim_size = dim_size
        self.reduce = reduce
    
    def forward_op(self, src, index, base=None):
        src = src.contiguous()
        index = broadcast(index, src, self.dim).contiguous()
        base = base.contiguous() if base is not None else None
        
        out = getattr(torch_scatter, f'scatter_{self.reduce}')(
            src, index, self.dim, base, self.dim_size)
        return out
    

def iter_data(prep_fn, *args, **kwargs):
    file_list = list((Path(__file__).parent.parent / 'data').glob('*.npy'))
    for file in file_list:
        src = torch.from_numpy(np.load(file))
        yield prep_fn(src, *args, **kwargs)


def scatter_prep_fn(src, dim, dim_size, with_base=False):
    
    x, y, z = src[:, :3].long().unbind(-1)
    x_idx, y_idx, z_idx = x - x.min(), y - y.min(), z - z.min()
    index = z_idx * (x.max() * y.max() + 1) + y_idx * (x.max() + 1) + x_idx
    index = (dim_size / (index.max()+1) * index).long()

    src = src[:, :4]
    if dim < 0:
        dim = src.dim() + dim
    for _ in range(dim):
        src = src.unsqueeze(0)
    
    base = None
    if with_base:
        base = src.new_ones(*src.shape[:dim], dim_size, *src.shape[dim + 1:]) * 1.123456789
    
    return src, index, base

