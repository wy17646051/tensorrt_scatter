import ctypes
import os.path as osp
import sys
from itertools import product
from typing import Dict, List

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