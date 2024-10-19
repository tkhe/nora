# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

from nora.layers import CNNBlock
from nora.layers import Conv2d
from nora.layers import DepthwiseSeparableConv2d
from nora.layers import ShapeSpec
from nora.layers import weight_init
from .base import Backbone

__all__ = ["CSPNeXt"]

CSPNEXT_DEFAULT_SETTING = [
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    [64, 128, 3, True, False],
    [128, 256, 6, True, False],
    [256, 512, 6, True, False],
    [512, 1024, 3, False, True],
]


class CSPNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "SiLU",
    ):
        super().__init__()

        hidden_dim = int(out_channels * expansion)
        self.conv1 = (DepthwiseSeparableConv2d if use_depthwise else Conv2d)(
            in_channels,
            hidden_dim,
            kernel_size=3,
            stride=1,
            norm=norm,
            activation=activation,
        )
        self.conv2 = DepthwiseSeparableConv2d(
            hidden_dim,
            out_channels,
            kernel_size,
            stride=1,
            norm=norm,
            activation=activation,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class ChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = Conv2d(channels, channels, kernel_size=1, bias=True, activation="HardSigmoid")

        weight_init.c2_msra_fill(self.fc)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)

        out = self.fc(out)
        return x * out


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        channel_attention: bool = False,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "SiLU",
    ):
        super().__init__()

        hidden_dim = int(out_channels * expand_ratio)
        self.main_conv = Conv2d(in_channels, hidden_dim, kernel_size=1, norm=norm, activation=activation)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(hidden_dim, hidden_dim, 1.0, add_identity, use_depthwise, norm=norm, activation=activation) for _ in range(num_blocks)])
        self.short_conv = Conv2d(in_channels, hidden_dim, kernel_size=1, norm=norm, activation=activation)
        self.attention = ChannelAttention(2 * hidden_dim) if channel_attention else None
        self.final_conv = Conv2d(2 * hidden_dim, out_channels, kernel_size=1, norm=norm, activation=activation)

    def forward(self, x):
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_short = self.short_conv(x)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.attention:
            x_final = self.attention(x_final)

        return self.final_conv(x_final)


class SPPFBottleneck(nn.Module):
    """
    "Spatial Pyramid Pooling - Fast version.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "SiLU",
    ):
        super().__init__()

        hidden_dim = int(in_channels * 0.5)
        padding = (kernel_size - 1) // 2
        self.conv1 = Conv2d(in_channels, hidden_dim, kernel_size=1, norm=norm, activation=activation)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = Conv2d(hidden_dim * 4, out_channels, kernel_size=1, norm=norm, activation=activation)

        for m in [self.conv1, self.conv2]:
            weight_init.c2_msra_fill(m)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        x = torch.cat([x, y1, y2, y3], dim=1)
        x = self.conv2(x)
        return x


class BasicStem(CNNBlock):
    def __init__(
        self,
        widen_factor: float,
        in_channels: int,
        out_channels: int,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "SiLU",
    ):
        hidden_dim = int(out_channels * widen_factor // 2)
        out_channels = int(out_channels * widen_factor)

        super().__init__(in_channels, out_channels, 2)

        self.conv1 = Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, norm=norm, activation=activation)
        self.conv2 = Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, norm=norm, activation=activation)
        self.conv3 = Conv2d(hidden_dim, out_channels, kernel_size=3, stride=1, norm=norm, activation=activation)

        for m in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(m)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CSPNeXtStage(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        expand_ratio: float,
        add_identity: bool,
        use_depthwise: bool,
        channel_attention: bool,
        use_spp: bool,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "SiLU",
    ):
        super().__init__(in_channels, out_channels, 2)

        self.conv = (DepthwiseSeparableConv2d if use_depthwise else Conv2d)(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm=norm,
            activation=activation,
        )
        self.spp = SPPFBottleneck(out_channels, out_channels, kernel_size=5, norm=norm, activation=activation) if use_spp else None
        self.csp = CSPLayer(
            out_channels,
            out_channels,
            expand_ratio,
            num_blocks,
            add_identity,
            use_depthwise,
            channel_attention,
            norm,
            activation,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.spp is not None:
            x = self.spp(x)
        x =  self.csp(x)
        return x


class CSPNeXt(Backbone):
    """
    support CSPNeXt proposed in RTMDet (https://arxiv.org/abs/2212.07784).
    """

    def __init__(
        self,
        deepen_factor: float,
        widen_factor: float,
        arch_setting: Optional[List[List]] = None,
        input_shape: Optional[ShapeSpec] = None,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        channel_attention: bool = True,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "SiLU",
        out_features: Optional[List[str]] = None,
        freeze_at: int = 0,
    ):
        """
        Args:
            deepen_factor (float): Depth multiplier, multiply number of 
                blocks in CSP layer by this amount.
            widen_factor (float): Width multiplier, multiply number of
                channels in each layer by this amount.
            arch_setting (list[list]): 
            input_shape (ShapeSpec, optional): shape of input image.
                If None, input image has 3 channels by default.
            use_depthwise (bool): Whether to use depthwise separable convolution.
            expand_ratio (float): Ratio to adjust the number of channels of the
                hidden layer.
            channel_attention (bool): Whether to add channel attention in each
                stage. Defaults to True.
            norm (str, dict): Normalization layer. Default: BN.
            activation (str, dict): Activation layer. Default: SiLU.
            out_features (list[int]): Output from which stages.
            freeze_at (int): Stages to be frozen (stop grad and set eval mode).
                0 means not freezing any parameters.
        """
        super().__init__()

        if arch_setting is None:
            arch_setting = CSPNEXT_DEFAULT_SETTING

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f) for f in out_features])
            arch_setting = arch_setting[:num_stages]

        self.stem = BasicStem(
            widen_factor=widen_factor,
            in_channels=input_shape.channels,
            out_channels=arch_setting[0][0],
            norm=norm,
            activation=activation,
        )
        cur_stride = 2
        self._out_feature_channels["stem"] = self.stem.out_channels
        self._out_feature_strides["stem"] = cur_stride

        self.stages = []
        for idx, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting, start=2):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)

            stage_name = f"stage{idx}"
            stage = CSPNeXtStage(
                in_channels,
                out_channels,
                num_blocks,
                expand_ratio,
                add_identity,
                use_depthwise,
                channel_attention,
                use_spp,
                norm,
                activation,
            )
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)
            cur_stride *= 2
            self._out_feature_channels[stage_name] = out_channels
            self._out_feature_strides[stage_name] = cur_stride

        self._out_features = out_features
        assert len(self._out_features)

        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, f"Available children: {', '.join(children)}"

        self.freeze(freeze_at)

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "CSPNeXt-tiny": cspnext_tiny,
            "CSPNeXt-small": cspnext_small,
            "CSPNeXt-medium": cspnext_medium,
            "CSPNeXt-large": cspnext_large,
            "CSPNeXt-xl": cspnext_xl,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for CSPNeXt.")

        return predefined[name](**kwargs)

    def freeze(self, freeze_at: int = 0):
        if freeze_at >= 1:
            self.stem.freeze()

        for idx, stage_name in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                stage = getattr(self, stage_name)
                stage.freeze()

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


def cspnext_tiny(**kwargs):
    return CSPNeXt(deepen_factor=0.167, widen_factor=0.375, **kwargs)


def cspnext_small(**kwargs):
    return CSPNeXt(deepen_factor=0.33, widen_factor=0.5, **kwargs)


def cspnext_medium(**kwargs):
    return CSPNeXt(deepen_factor=0.67, widen_factor=0.75, **kwargs)


def cspnext_large(**kwargs):
    return CSPNeXt(deepen_factor=1.0, widen_factor=1.0, **kwargs)


def cspnext_xl(**kwargs):
    return CSPNeXt(deepen_factor=1.33, widen_factor=1.25, **kwargs)
