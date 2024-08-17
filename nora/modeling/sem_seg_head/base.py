from abc import ABC
from abc import abstractmethod
from typing import Dict

import torch
import torch.nn as nn

__all__ = ["SemSegHead"]


class SemSegHead(nn.Module, ABC):
    """
    Abstract base class for semantic segmentation head.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, features: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError
