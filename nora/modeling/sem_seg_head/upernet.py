import math
from typing import Dict
from typing import List
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nora.layers import Conv2d
from nora.layers import PPM
from nora.layers import ShapeSpec
from nora.layers import weight_init
from nora.modeling.sem_seg_head import SemSegHead

__all__ = ["UPerNetHead"]


class UPerNetHead(SemSegHead):
    """
    support UPerNetHead (https://arxiv.org/abs/1807.10221).
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        in_features: List[int],
        *,
        num_classes: int,
        channels: int,
        pool_scales: List[int] = [1, 2, 3, 6],
        dropout: float = 0.1,
        ignore_value: int = -1,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "ReLU",
        loss_sem_seg: Union[nn.Module, List[nn.Module]],
        align_corners: bool = False,
    ):
        """
        Args:
            pool_scales (list[int]): Pooling scales used in PPM.
            dropout (float): Ratio of dropout layer.
            norm (str or dict): Normalization layer.
            activation (str or dict): Activation layer.
            loss_sem_seg (nn.Module or list[nn.Module]): loss(es) for 
            align_corners (bool): Argument of F.interpolate.
        """
        super().__init__()

        strides = [input_shape[f].stride for f in in_features]
        self.in_channels = [input_shape[f].channels for f in in_features]
        self.align_corners = align_corners
        self.in_features = in_features
        self.ignore_value = ignore_value

        lateral_convs = []
        output_convs = []
        for idx, in_channels in enumerate(self.in_channels):
            if idx == len(self.in_channels) - 1:
                lateral_conv = PPM(pool_scales, self.in_channels[-1], channels, norm, activation, align_corners)
                output_conv = None
            else:
                lateral_conv = Conv2d(in_channels, channels, kernel_size=1, norm=norm, activation=activation)
                output_conv = Conv2d(channels, channels, kernel_size=3, stride=1, norm=norm, activation=activation)
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"fpn_lateral{stage}", lateral_conv)
            if output_conv is not None:
                self.add_module(f"fpn_output{stage}", output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.conv = Conv2d(
            len(self.in_channels) * channels,
            channels,
            kernel_size=3,
            stride=1,
            norm=norm,
            activation=activation,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.predictor = Conv2d(channels, num_classes, kernel_size=1)

        if isinstance(loss_sem_seg, nn.Module):
            assert loss_sem_seg.name is not None

            self.loss_sem_seg = loss_sem_seg
        elif isinstance(loss_sem_seg, (tuple, list)):
            assert all(loss.name is not None for loss in loss_sem_seg)

            self.loss_sem_seg = nn.ModuleList(loss_sem_seg)
        else:
            raise TypeError(f"loss_sem_seg must be a nn.Module or sequence of nn.Module, but got {type(loss_sem_seg)}")

    def forward(self, bottom_up_features: Dict[str, torch.Tensor]):
        results = []

        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(prev_features)

        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx > 0:
                features = bottom_up_features[self.in_features[-idx - 1]]
                top_down_features = F.interpolate(
                    prev_features,
                    size=features.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                results.insert(0, output_conv(prev_features))

        for i in range(1, len(results)):
            results[i] = F.interpolate(
                results[i],
                size=results[0].shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        results = torch.cat(results, dim=1)
        results = self.conv(results)
        results = self.dropout(results)
        results = self.predictor(results)
        return results

    def loss(self, predictions, targets):
        if predictions.size()[-2:] != targets.size()[-2:]:
            predictions = F.interpolate(
                predictions,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        losses = {}
        if isinstance(self.loss_sem_seg, nn.ModuleList):
            for loss in self.loss_sem_seg:
                losses[loss.name] = loss(predictions, targets)
        else:
            losses[loss.name] = self.loss_sem_seg(predictions, targets)

        return losses

    def inference(self, predictions, image_size):
        return F.interpolate(predictions, size=image_size, mode="bilinear", align_corners=False)
