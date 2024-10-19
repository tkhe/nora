# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List

import torch
import torch.nn as nn

from nora.layers import ShapeSpec

__all__ = ["Backbone"]


class Backbone(nn.Module, ABC):
    """
    Abstract base class for backbones.
    """

    def __init__(self):
        super().__init__()

        self._out_features: List[str] = []
        self._out_feature_channels: Dict[str, int] = {}
        self._out_feature_strides: Dict[str, int] = {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        raise NotImplementedError

    @property
    def padding_constraints(self) -> int:
        return {
            "size_divisiblity": 0,
            "square_size": 0,
        }

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at: int = -1):
        raise NotImplementedError
