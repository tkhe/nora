# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC
from abc import abstractmethod

import torch.nn as nn

from .norm import FrozenBatchNorm2d

__all__ = ["CNNBlock"]


class CNNBlock(nn.Module, ABC):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batch_norm(self)
        return self
