import torch
import torch.nn as nn
from torch.nn import Parameter


class BaseModel(nn.Module):                 
    
    def __init__(self, params, device):
        super(BaseModel, self).__init__()
        self.p = params
        self.device = device








