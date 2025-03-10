import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class DenseBlock(nn.Module):

    def __init__(self, input_size: int, hidden_layers: list, output_len: int):
        super(DenseBlock, self).__init__()
        
