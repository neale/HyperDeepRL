#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import *


class BaseNet:
    def __init__(self):
        pass

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def ortho_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def kaiming_init(layer, w_scale=1.0):
    nn.init.kaiming_uniform_(
        layer.weight.data,
        mode='fan_out', 
        nonlinearity='relu')
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for "\
            "tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)


def no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def hyper_uniform(layer):
    fan_in, fan_out = calculate_fan_in_and_fan_out(layer.weight.data)
    #var_e = 1
    var_e = 1/12.
    a = 2
    b = fan_in * fan_out * var_e
    x = a / b
    no_grad_uniform_(
        layer.weight.data,
        -(3 * x)**.5,
         (3 * x)**.5)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def hyper_normal(layer):
    fan_in, fan_out = calculate_fan_in_and_fan_out(layer.weight.data)
    var_e = 1
    a = 2
    b = fan_in * fan_out * var_e
    x = a / b
    no_grad_normal_(layer.weight.data, 0, x)
    nn.init.constant_(layer.bias.data, 0)
    return layer

