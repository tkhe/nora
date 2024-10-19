from abc import ABC
from abc import abstractmethod

import torch

__all__ = ["BoxCoder"]


class BoxCoder(ABC):
    @abstractmethod
    def get_deltas(self, src_boxes: torch.Tensor, target_boxes: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def apply_deltas(self, deltas: torch.Tensor, boxes: torch.Tensor):
        raise NotImplementedError
