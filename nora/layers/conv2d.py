# Copyright (c) Facebook, Inc. and its affiliates.

import warnings
from typing import Dict
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

from .activation import get_activation
from .misc import check_if_dynamo_compiling
from .norm import get_norm

__all__ = [
    "Conv2d",
    "DepthwiseSeparableConv2d",
]


class Conv2d(nn.Module):
    """
    Enhance `torch.nn.Conv2d` to support more features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str] = "auto",
        dilation: int = 1,
        groups: int = 1,
        bias: Union[bool, str] = "auto",
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        norm: Optional[Union[str, Dict]] = None,
        activation: Optional[Union[str, Dict]] = None,
    ):
        super().__init__()

        if padding == "auto":
            padding = (kernel_size - 1) // 2 * dilation

        if bias == "auto":
            bias = not norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.norm = get_norm(norm, out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        if not torch.jit.is_scripting():
            # Dynamo doesn't support context managers yet
            is_dynamo_compiling = check_if_dynamo_compiling()
            if not is_dynamo_compiling:
                with warnings.catch_warnings(record=True):
                    if x.numel() == 0 and self.training:
                        # https://github.com/pytorch/pytorch/issues/12013
                        assert not isinstance(self.norm, torch.nn.SyncBatchNorm), "SyncBatchNorm does not support empty inputs!"

        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise separable convolution.

    See https://arxiv.org/pdf/1704.04861.pdf for details.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str] = "auto",
        dilation: int = 1,
        bias: Union[bool, str] = "auto",
        norm: Optional[Union[str, Dict]] = None,
        activation: Optional[Union[str, Dict]] = None,
        dw_norm: Union[Dict, str] = "default",
        dw_activation: Union[Dict, str] = "default",
        pw_norm: Union[Dict, str] = "default",
        pw_activation: Union[Dict, str] = "default",
    ):
        """
        Args:
            norm (str or dict): Default normalization layer for both depthwise conv
                and pointwise conv.
            activation (str or dict): Default activation layer for both depthwise conv
                and pointwise conv.
            dw_norm (str or dict): Normalization layer for depthwise conv. If it is `default`,
                it will be the same as `norm`.
            dw_activation (str or dict): Activation layer for depthwise conv. If it is `default`,
                it will be the same as `activation`.
            pw_norm (str or dict): Normalization layer for pointwise conv. If it is `default`,
                it will be the same as `norm`.
            pw_activation (str or dict): Activation layer for pointwise conv. If it is `default`,
                it will be the same as `activation`.
        """
        super().__init__()

        if dw_norm == "default":
            dw_norm = norm

        if dw_activation == "default":
            dw_activation = activation

        if pw_norm == "default":
            pw_norm = norm

        if pw_activation == "default":
            pw_activation = activation

        dw_bias = not dw_norm if bias == "auto" else bias
        pw_bias = not pw_norm if bias == "auto" else bias

        self.dw_conv = Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=dw_bias, norm=dw_norm, activation=dw_activation)
        self.pw_conv = Conv2d(in_channels, out_channels, kernel_size=1, bias=pw_bias, norm=pw_norm, activation=pw_activation)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x
