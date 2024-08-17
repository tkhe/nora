from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

from nora.layers import CNNBlock
from nora.layers import Conv2d
from nora.layers import DropPath
from nora.layers import ShapeSpec
from nora.layers import get_activation
from nora.layers import get_norm
from .base import Backbone

__all__ = ["ConvNeXt"]


class BasicStem(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: Union[str, Dict] = "LN2d",
    ):
        super().__init__(in_channels, out_channels, 4)

        self.conv = Conv2d(in_channels, out_channels, kernel_size=4, stride=4, padding=0)
        self.norm = get_norm(norm, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class ConvNeXtBlock(CNNBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        mlp_ratio: float = 4.0,
        norm: Union[str, Dict] = "LN2d",
        activation: Union[str, Dict] = "GELU",
        layer_scale_init_value: float =1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__(in_channels, out_channels, stride)

        if self.stride == 2:
            self.downsample = nn.Sequential(
                get_norm(norm, in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            )
        else:
            self.downsample = None
            assert in_channels == out_channels

        hidden_dim = int(out_channels * mlp_ratio)

        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, groups=out_channels)
        self.norm = get_norm(norm, out_channels)
        self.pwconv1 = nn.Linear(out_channels, hidden_dim)
        self.activation = get_activation(activation)
        self.pwconv2 = nn.Linear(hidden_dim, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        if self.stride == 2:
            x = self.downsample(x)

        input = x
        x = self.dwconv(x)
        # [N, C, H, W] -> [N, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x, channel_last=True)
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # [N, H, W, C] -> [N, C, H, W]
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(Backbone):
    """
    support ConvNeXt (https://arxiv.org/abs/2201.03545).
    """

    def __init__(
        self,
        depths: List[int],
        channels: List[int],
        input_shape: Optional[ShapeSpec] = None,
        norm: Union[str, Dict] = "LN2d",
        activation: Union[str, Dict] = "GELU",
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        out_features: Optional[List[str]] = None,
        freeze_at: int = -1,
    ):
        """
        Args:
            depths (List[int]): number of blocks in each stage.
            channels (List[int]): number of channels in each stage.
            input_shape (ShapeSpec): input shape.
            norm (str or dict): normalization layer.
            activation (str or dict): activation layer.
            drop_path_rate (float): stochastic depth rate.
            layer_scale_init_value (float): init value for layer scale.
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim.
            out_features (List[str]): name of the layers whose outputs should be returned in forward.
            freeze_at (int): freeze the backbone at the given stage.
                -1 means not freezing any parameters.
        """
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features])
            depths = depths[:num_stages]
            channels = channels[:num_stages]

        self.stem = BasicStem(input_shape.channels, channels[0], norm=norm)
        cur_stride = self.stem.stride
        self._out_feature_strides["stem"] = cur_stride
        self._out_feature_channels["stem"] = self.stem.out_channels

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = []
        for stage_idx, depth in enumerate(depths):
            stage = []
            name = f"stage{stage_idx + 2}"
            stride = 2 if stage_idx > 0 else 1
            out_channels = channels[stage_idx]
            stage.append(
                ConvNeXtBlock(
                    channels[stage_idx] if stride == 1 else channels[stage_idx - 1],
                    channels[stage_idx],
                    stride=stride,
                    mlp_ratio=mlp_ratio,
                    norm=norm,
                    activation=activation,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_path=dpr[sum(depths[:stage_idx])],
                )
            )
            for block_idx in range(1, depth):
                stage.append(
                    ConvNeXtBlock(
                        out_channels,
                        out_channels,
                        stride=1,
                        mlp_ratio=mlp_ratio,
                        norm=norm,
                        activation=activation,
                        layer_scale_init_value=layer_scale_init_value,
                        drop_path=dpr[sum(depths[:stage_idx]) + block_idx],
                    )
                )
            self.add_module(name, nn.Sequential(*stage))
            self.stages.append(name)

            cur_stride *= stride
            self._out_feature_channels[name] = out_channels
            self._out_feature_strides = cur_stride

        self._out_features = out_features
        assert len(self._out_features)

        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, f"Available children: {', '.join(children)}"

        self.freeze(freeze_at)

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "ConvNeXt-tiny": convnext_tiny,
            "ConvNeXt-small": convnext_small,
            "ConvNeXt-base": convnext_base,
            "ConvNeXt-large": convnext_large,
            "ConvNeXt-xlarge": convnext_xlarge,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for ConvNeXt.")

        return predefined[name](**kwargs)

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


def convnext_tiny(**kwargs):
    return ConvNeXt([3, 3, 9, 3], [96, 192, 384, 768], drop_path_rate=0.1, **kwargs)


def convnext_small(**kwargs):
    return ConvNeXt([3, 3, 27, 3], [96, 192, 384, 768], drop_path_rate=0.4, **kwargs)


def convnext_base(**kwargs):
    return ConvNeXt([3, 3, 27, 3], [128, 256, 512, 1024], drop_path_rate=0.5, **kwargs)


def convnext_large(**kwargs):
    return ConvNeXt([3, 3, 27, 3], [192, 384, 768, 1536], drop_path_rate=0.5, **kwargs)


def convnext_xlarge(**kwargs):
    return ConvNeXt([3, 3, 27, 3], [256, 512, 1024, 2048], drop_path_rate=0.5, **kwargs)
