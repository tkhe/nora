# Copyright (c) Facebook, Inc. and its affiliates.

import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import torch
import torch.nn as nn
from torch.utils import checkpoint

from nora.layers import CNNBlock
from nora.layers import Conv2d
from nora.layers import get_activation
from nora.layers import weight_init
from nora.layers.shape_spec import ShapeSpec
from .base import Backbone
from .resnet import BasicStem

__all__ = ["ResNeXt"]


class Bottleneck(CNNBlock):
    """
    Bottleneck block for ResNeXt.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        stride_in_1x1: bool = False,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
        groups: int = 1,
        base_width: int = 4,
        base_channels: int = 64,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "ReLU",
        use_checkpoint: bool = False,
    ):
        super().__init__(in_channels, out_channels, stride)

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        bottleneck_channels = out_channels // 4

        if groups == 1:
            width = bottleneck_channels
        else:
            width = math.floor(bottleneck_channels * (base_width / base_channels)) * groups

        self.use_checkpoint = use_checkpoint

        self.conv1 = Conv2d(in_channels, width, kernel_size=1, stride=stride_1x1, norm=norm, activation=activation)
        self.conv2 = Conv2d(width, width, kernel_size=3, stride=stride_3x3, dilation=dilation, groups=groups, norm=norm, activation=activation)
        self.conv3 = Conv2d(width, out_channels, kernel_size=1, stride=1, norm=norm)
        self.downsample = downsample
        self.activation = get_activation(activation)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x: torch.Tensor):

        def _inner_forward(x):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                shortcut = self.downsample(x)
            else:
                shortcut = x

            out += shortcut
            return out

        if self.use_checkpoint and x.requires_grad:
            out = checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.activation(out)
        return out


class ResNeXt(Backbone):
    """
    support ResNeXt (https://arxiv.org/abs/1611.05431).
    """

    def __init__(
        self,
        stem_class,
        block_class,
        num_blocks: List[int],
        out_channels_per_stage: List[int],
        input_shape: Optional[ShapeSpec] = None,
        stem_out_channels: int = 64,
        strides: List[int] = [1, 2, 2, 2],
        dilations: List[int] = [1, 1, 1, 1],
        groups: int = 1,
        base_width: int = 4,
        base_channels: int = 64,
        stride_in_1x1: bool = False,
        use_checkpoint: bool = False,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "ReLU",
        out_features: Optional[List[str]] = None,
        freeze_at: int = 0,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        if out_features is not None:
            num_stages = max([{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features])
            num_blocks = num_blocks[:num_stages]
            out_channels_per_stage = out_channels_per_stage[:num_stages]
            strides = strides[:num_stages]
            dilations = dilations[:num_stages]

        self.stem = stem_class(input_shape.channels, stem_out_channels, norm=norm, activation=activation)
        cur_stride = self.stem.stride
        self._out_feature_strides["stem"] = cur_stride
        self._out_feature_channels["stem"] = self.stem.out_channels

        in_channels = self.stem.out_channels
        self.stages = []
        for idx, (num_block, out_channels, stride, dilation) in enumerate(zip(num_blocks, out_channels_per_stage, strides, dilations), start=2):
            name = f"res{idx}"
            stage = ResNeXt.make_stage(
                block_class,
                num_block,
                in_channels,
                out_channels,
                stride,
                stride_in_1x1,
                dilation,
                groups,
                base_width,
                base_channels,
                norm,
                activation,
                use_checkpoint,
            )
            self.add_module(name, stage)
            self.stages.append(name)
            self._out_feature_channels[name] = out_channels
            cur_stride *= stride
            self._out_feature_strides[name] = cur_stride
            in_channels = out_channels

        self._out_features = out_features
        assert len(self._out_features)

        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, f"Available children: {', '.join(children)}"

        self.freeze(freeze_at)

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "ResNeXt-50-32x4d": resnext50_32x4d,
            "ResNeXt-101-32x4d": resnext101_32x4d,
            "ResNeXt-101-32x8d": resnext101_32x8d,
            "ResNeXt-101-64x4d": resnext101_64x4d,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for ResNet.")

        return predefined[name](**kwargs)

    @staticmethod
    def make_stage(
        block_class: Type[Bottleneck],
        num_blocks,
        in_channels: int,
        out_channels: int,
        stride: int,
        stride_in_1x1: bool = False,
        dilation: int = 1,
        groups: int = 1,
        base_width: int = 4,
        base_channels: int = 64,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "ReLU",
        use_checkpoint: bool = False,
    ):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, norm=norm)
            weight_init.c2_msra_fill(downsample)

        layers = [
            block_class(
                in_channels,
                out_channels,
                stride=stride,
                stride_in_1x1=stride_in_1x1,
                downsample=downsample,
                dilation=dilation,
                groups=groups,
                base_width=base_width,
                base_channels=base_channels,
                norm=norm,
                activation=activation,
                use_checkpoint=use_checkpoint,
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                block_class(
                    out_channels,
                    out_channels,
                    stride=1,
                    stride_in_1x1=stride_in_1x1,
                    dilation=dilation,
                    groups=groups,
                    base_width=base_width,
                    base_channels=base_channels,
                    norm=norm,
                    activation=activation,
                    use_checkpoint=use_checkpoint,
                )
            )
        return nn.Sequential(*layers)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.stem.freeze()

        for idx, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                stage = getattr(self, stage_name)
                for block in stage.children():
                    block.freeze()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}

        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if stage_name in self._out_features:
                outputs[stage_name] = x

        return outputs


def resnext50_32x4d(**kwargs):
    return ResNeXt(BasicStem, Bottleneck, [3, 4, 6, 3], [256, 512, 1024, 2048], groups=32, base_width=4, **kwargs)


def resnext101_32x4d(**kwargs):
    return ResNeXt(BasicStem, Bottleneck, [3, 4, 23, 3], [256, 512, 1024, 2048], groups=32, base_width=4, **kwargs)


def resnext101_32x8d(**kwargs):
    return ResNeXt(BasicStem, Bottleneck, [3, 4, 23, 3], [256, 512, 1024, 2048], groups=32, base_width=8, **kwargs)


def resnext101_64x4d(**kwargs):
    return ResNeXt(BasicStem, Bottleneck, [3, 4, 23, 3], [256, 512, 1024, 2048], groups=64, base_width=4, **kwargs)
