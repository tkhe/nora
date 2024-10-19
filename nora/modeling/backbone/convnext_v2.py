from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

from nora.layers import CNNBlock
from nora.layers import DropPath
from nora.layers import ShapeSpec
from nora.layers import get_activation
from nora.layers import get_norm
from .base import Backbone
from .convnext import BasicStem

__all__ = ["ConvNeXtV2"]


class GRN(nn.Module):
    """
    Global Response Normalization Module.

    Proposed in ConvNeXt v2 (http://arxiv.org/abs/2301.00808).
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()

        self.channels = channels
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        x = self.gamma * (x * nx) + self.beta + x
        return x


class ConvNeXtV2Block(CNNBlock):
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
        self.grn = GRN(hidden_dim)
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
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # [N, H, W, C] -> [N, C, H, W]
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(Backbone):
    """
    support ConvNeXt v2 (http://arxiv.org/abs/2301.00808).
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
                ConvNeXtV2Block(
                    channels[stage_idx] if stride == 1 else channels[stage_idx - 1],
                    out_channels,
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
                    ConvNeXtV2Block(
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
            "ConvNeXt-V2-atto": convnext_v2_atto,
            "ConvNeXt-V2-femto": convnext_v2_femto,
            "ConvNeXt-V2-pico": convnext_v2_pico,
            "ConvNeXt-V2-nano": convnext_v2_nano,
            "ConvNeXt-V2-tiny": convnext_v2_tiny,
            "ConvNeXt-V2-small": convnext_v2_small,
            "ConvNeXt-V2-base": convnext_v2_base,
            "ConvNeXt-V2-large": convnext_v2_large,
            "ConvNeXt-V2-xlarge": convnext_v2_xlarge,
            "ConvNeXt-V2-huge": convnext_v2_huge,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for ConvNeXt-V2.")

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


def convnext_v2_atto(**kwargs):
    return ConvNeXtV2([2, 2, 6, 2], [40, 80, 160, 320], drop_path_rate=0.1, **kwargs)


def convnext_v2_femto(**kwargs):
    return ConvNeXtV2([2, 2, 6, 2], [48, 96, 192, 384], drop_path_rate=0.1, **kwargs)


def convnext_v2_pico(**kwargs):
    return ConvNeXtV2([2, 2, 6, 2], [64, 128, 256, 512], drop_path_rate=0.1, **kwargs)


def convnext_v2_nano(**kwargs):
    return ConvNeXtV2([2, 2, 8, 2], [80, 160, 320, 640], drop_path_rate=0.1, **kwargs)


def convnext_v2_tiny(**kwargs):
    return ConvNeXtV2([3, 3, 9, 3], [96, 192, 384, 768], drop_path_rate=0.1, **kwargs)


def convnext_v2_small(**kwargs):
    return ConvNeXtV2([3, 3, 27, 3], [96, 192, 384, 768], drop_path_rate=0.1, **kwargs)


def convnext_v2_base(**kwargs):
    return ConvNeXtV2([3, 3, 27, 3], [128, 256, 512, 1024], drop_path_rate=0.1, **kwargs)


def convnext_v2_large(**kwargs):
    return ConvNeXtV2([3, 3, 27, 3], [192, 384, 768, 1536], drop_path_rate=0.1, **kwargs)


def convnext_v2_xlarge(**kwargs):
    return ConvNeXtV2([3, 3, 27, 3], [256, 512, 1024, 2048], drop_path_rate=0.1, **kwargs)


def convnext_v2_huge(**kwargs):
    return ConvNeXtV2([3, 3, 27, 3], [352, 704, 1408, 2816], drop_path_rate=0.1, **kwargs)
