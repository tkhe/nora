# Copyright (c) Facebook, Inc. and its affiliates.

from functools import wraps

import torch.nn as nn

from .conv2d import Conv2d
from .conv2d import DepthwiseSeparableConv2d

__all__ = [
    "bias_fill",
    "c2_msra_fill",
    "c2_xavier_fill",
    "constant_fill",
    "init_conv2d",
    "normal_fill",
]


def init_conv2d(func):
    @wraps(func)
    def deco(module, *args, **kwargs):
        if isinstance(module, Conv2d):
            func(module.conv, *args, **kwargs)
        elif isinstance(module, DepthwiseSeparableConv2d):
            func(module.dw_conv, *args, **kwargs)
            func(module.pw_conv, *args, **kwargs)
        else:
            func(module, *args, **kwargs)

    return deco


@init_conv2d
def c2_msra_fill(module, a: float = 0, mode: str = "fan_out", nonlinearity: str = "relu"):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_conv2d
def c2_xavier_fill(module, a: float = 1, mode: str = "fan_in", nonlinearity: str = "leaky_relu"):
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_conv2d
def normal_fill(module, mean: float = 0, std: float = 0.01):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)


@init_conv2d
def bias_fill(module, val: float = 0):
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, val)


@init_conv2d
def constant_fill(module, val: float = 1, bias: float = 0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val=val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)
