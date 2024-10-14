from typing import Dict
from typing import List
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2d import Conv2d

__all__ = ["PPM"]


class PPM(nn.Module):
    """
    Pyramid Pooling Module.
    """

    def __init__(
        self,
        pool_scales: List[int],
        in_channels: int,
        channels: int,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "ReLU",
        align_corners: bool = False,
    ):
        """
        Args:
            pool_scales (list[int]): Pooling scales used in PPM.
            in_channels (int): Input channels.
            channels (int): Output channels for each branch.
            norm (str or dict): Normalization layer.
            activation (str or dict): Activation layer.
            align_corners (bool): Argument of F.interpolate.
        """
        super().__init__()

        poolers = []
        for pool_scale in pool_scales:
            layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                Conv2d(in_channels, channels, kernel_size=1, norm=norm, activation=activation),
            )
            poolers.append(layer)

        self.poolers = nn.ModuleList(poolers)
        self.conv = Conv2d(
            in_channels + len(pool_scales) * channels,
            channels,
            kernel_size=3,
            stride=1,
            norm=norm,
            activation=activation,
        )
        self.align_corners = align_corners

    def forward(self, x):
        outs = [x]
        for pool in self.poolers:
            out = pool(x)
            out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners)
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        outs = self.conv(outs)
        return outs
