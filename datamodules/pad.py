import torch
from functools import partial

def pad(tensor, length):
    if tensor.shape[-1] == length:
        return tensor
    p = torch.zeros((1, length - tensor.shape[-1]))
    tensor = torch.cat([tensor, p], dim=-1)
    return tensor

def make_pad_function(length):
    return partial(pad, length=length)
    