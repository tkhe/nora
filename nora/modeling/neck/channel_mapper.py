# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

from nora.layers import Conv2d
from nora.layers import ShapeSpec
from nora.layers import weight_init
from .base import Neck

__all__ = ["ChannelMapper"]


class ChannelMapper(Neck):
    """
    Channel Mapper for reduce/increase channels of backbone features.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        in_features: List[str],
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "ReLU",
        num_outs: Optional[int] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.extra_convs = None

        if num_outs is None:
            num_outs = len(in_features)

        self.convs = nn.ModuleList()
        for in_channels in [input_shape[f].channels for f in in_features]:
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                norm=norm,
                activation=activation,
            )
            weight_init.c2_msra_fill(conv)
            self.convs.append(conv)

        if num_outs > len(in_features):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_features), num_outs):
                if i == len(in_features):
                    in_channels = input_shape[in_features[-1]].channels
                else:
                    in_channels = out_channels
                conv = Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    dilation=dilation,
                    groups=groups,
                    norm=norm,
                    activation=activation,
                )
                weight_init.c2_msra_fill(conv)
                self.extra_convs.append(conv)

        strides = [input_shape[f].stride for f in in_features]
        if num_outs > len(in_features):
            for _ in range(len(in_features), num_outs):
                strides.append(max(strides) * 2)

        self._out_features = [f"p{int(math.log2(s))}" for s in strides]
        self._out_feature_strides = {f: s for f, s in zip(self._out_features, strides)}
        self._out_feature_channels = {f: out_channels for f in self._out_features}

    def forward(self, bottom_up_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        results = []

        for idx, f in enumerate(self.in_features):
            results.append(self.convs[idx](bottom_up_features[f]))

        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    results.append(self.extra_convs[i](bottom_up_features[self.in_features[-1]]))
                else:
                    results.append(self.extra_convs[i](results[-1]))

        return {f: res for f, res in zip(self._out_features, results)}

    @property
    def padding_constraints(self):
        return {
            "size_divisibility": max(self._out_feature_strides.values()),
            "square_size": 0,
        }
