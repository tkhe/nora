# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Dict
from typing import List
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nora.layers import Conv2d
from nora.layers import DepthwiseSeparableConv2d
from nora.layers import FrozenBatchNorm2d
from nora.layers import ShapeSpec
from nora.modeling.backbone.cspnext import CSPLayer
from .base import Neck

__all__ = ["CSPNeXtPAFPN"]


class CSPNeXtPAFPN(Neck):
    """
    Path Aggregation Network with CSPNeXt blocks. Proposed in (https://arxiv.org/abs/2212.07784).
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        in_features: List[str],
        out_channels: int,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "SiLU",
        freeze_all: bool = False,
    ):
        """
        Args:
            input_shape (Dict[str, ShapeSpec]): input shape.
            in_features (List[str]): input feature names.
            out_channels (int): number of output channels (used at each scale)
            deepen_factor (float): depth multiplier, multiply number of
                blocks in CSP layer by this amount.
            widen_factor (float): width multiplier, multiply number of
                channels in each layer by this amount.
            num_csp_blocks (int): Number of bottlenecks in CSPLayer.
            use_depthwise (bool): Whether to use depthwise separable convolution in blocks.
            expand_ratio (float): Ratio to adjust the number of channels of the hidden layer.
            norm (str or dict): normalization layer.
            activation (str or dict): activation layer.
            freeze_all (bool): whether to freeze all the layers of the neck.
        """
        super().__init__()

        in_channels = [int(input_shape[f].channels * widen_factor) for f in in_features]
        out_channels = int(out_channels * widen_factor)
        num_csp_blocks = round(num_csp_blocks * deepen_factor)

        strides = [input_shape[f].stride for f in in_features]
        self._out_features = [f"p{int(math.log2(s))}" for s in strides]
        self._out_feature_strides = {f: s for f, s in zip(self._out_features, strides)}
        self._out_feature_channels = {f: out_channels for f in self._out_features}

        self.in_features = in_features

        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            if idx == len(in_channels) - 1:
                layer = (DepthwiseSeparableConv2d if use_depthwise else Conv2d)(
                    in_channels[idx],
                    in_channels[idx - 1],
                    kernel_size=1,
                    stride=1,
                    norm=norm,
                    activation=activation,
                )
            else:
                layer = nn.Identity()

            self.reduce_layers.append(layer)

        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            if idx == 1:
                layer = CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    expand_ratio=expand_ratio,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    norm=norm,
                    activation=activation,
                )
            else:
                layer = nn.Sequential(
                    CSPLayer(
                        in_channels[idx - 1] * 2,
                        in_channels[idx - 1],
                        expand_ratio=expand_ratio,
                        num_blocks=num_csp_blocks,
                        add_identity=False,
                        norm=norm,
                        activation=activation,
                    ),
                    (DepthwiseSeparableConv2d if use_depthwise else Conv2d)(
                        in_channels[idx - 1],
                        in_channels[idx - 2],
                        kernel_size=1,
                        stride=1,
                        norm=norm,
                        activation=activation,
                    ),
                )
            self.top_down_layers.append(layer)

        self.downsample_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            layer = (DepthwiseSeparableConv2d if use_depthwise else Conv2d)(
                in_channels[idx],
                in_channels[idx],
                kernel_size=3,
                stride=2,
                norm=norm,
                activation=activation,
            )
            self.downsample_layers.append(layer)

        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            layer = CSPLayer(
                in_channels[idx] * 2,
                in_channels[idx + 1],
                num_blocks=num_csp_blocks,
                expand_ratio=expand_ratio,
                add_identity=False,
                norm=norm,
                activation=activation,
            )
            self.bottom_up_layers.append(layer)

        self.out_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            layer = (DepthwiseSeparableConv2d if use_depthwise else Conv2d)(
                in_channels[idx],
                out_channels,
                kernel_size=3,
                stride=1,
                norm=norm,
                activation=activation,
            )
            self.out_layers.append(layer)

        if freeze_all:
            self.freeze()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batch_norm(self)
        return self

    @property
    def padding_constraints(self):
        return {
            "size_divisibility": max(self._out_feature_strides.values()),
            "square_size": 0,
        }

    def forward(self, bottom_up_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = [bottom_up_features[f] for f in self.in_features]

        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_features)):
            reduce_outs.append(self.reduce_layers[idx](features[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_features) - 1, 0, -1):
            feature_high = inner_outs[0]
            feature_low = reduce_outs[idx - 1]
            upsample_feature = F.interpolate(feature_high, scale_factor=2.0, mode="nearest")
            top_down_layer_inputs = torch.cat([upsample_feature, feature_low], 1)
            inner_out = self.top_down_layers[len(self.in_features) - 1 - idx](top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_features) - 1):
            feature_low = outs[-1]
            feature_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feature_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feature_high], 1))
            outs.append(out)

        # out layers
        results = []
        for idx in range(len(self._out_features)):
            results.append(self.out_layers[idx](outs[idx]))

        assert len(results) == len(self._out_features)

        return {f: res for f, res in zip(self._out_features, results)}
