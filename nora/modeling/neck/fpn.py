# Copyright (c) Facebook, Inc. and its affiliates.

import math
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nora.layers import Conv2d
from nora.layers import ShapeSpec
from nora.layers import weight_init
from .base import Neck

__all__ = [
    "FPN",
    "FPNTopBlock",
    "LastLevelMaxPool",
    "LastLevelP6P7",
]


class FPNTopBlock(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @property
    def in_feature(self) -> str:
        return self._in_feature


class LastLevelMaxPool(FPNTopBlock):
    def __init__(self, in_feature: str = "p5"):
        super().__init__()

        self._num_levels = 1
        self._in_feature = in_feature

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(FPNTopBlock):
    def __init__(self, in_channels, out_channels, in_feature: str = "p5"):
        super().__init__()

        self._num_levels = 2
        self._in_feature = in_feature

        self.p6 = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class FPN(Neck):
    """
    support FPN (https://arxiv.org/abs/1612.03144).
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        in_features: List[str],
        out_channels: int,
        norm: Union[str, Dict] = "",
        activation: Union[str, Dict] = "",
        top_block: Optional[FPNTopBlock] = None,
        fuse_type: str = "sum",
    ):
        super().__init__()

        assert in_features, in_features
        assert fuse_type in {"sum", "avg"}

        strides = [input_shape[f].stride for f in in_features]
        in_channels_per_feature = [input_shape[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)

        lateral_convs = []
        output_convs = []
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_conv = Conv2d(in_channels, out_channels, kernel_size=1, norm=norm, activation=activation)
            output_conv = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, norm=norm, activation=activation)
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"fpn_lateral{stage}", lateral_conv)
            self.add_module(f"fpn_output{stage}", output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.top_block = top_block
        self.in_features = tuple(in_features)

        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in strides}
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._fuse_type = fuse_type

    def forward(self, bottom_up_features: Dict[str, torch.Tensor]):
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx > 0:
                features = bottom_up_features[self.in_features[-idx - 1]]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results)

        return {f: res for f, res in zip(self._out_features, results)}

    @property
    def padding_constraints(self):
        return {
            "size_divisibility": self._size_divisibility,
            "square_size": 0,
        }


def _assert_strides_are_log2_contiguous(strides):
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], f"Strides {stride} {strides[i - 1]} are not log2 contiguous"
