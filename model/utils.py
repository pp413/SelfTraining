import warnings
import time

import torch
# import torch_sparse
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

import sys


def get_param(shape):               
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param
