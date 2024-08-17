from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List

import torch
import torch.nn as nn

from nora.layers import move_device_like
from nora.structures import ImageList

__all__ = ["BaseModel"]


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    def with_neck(self) -> bool:
        return hasattr(self, "neck") and self.neck is not None

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    @abstractmethod
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        raise NotImplementedError

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        padding_constraints = {}
        padding_constraints.update(self.backbone.padding_constraints)
        if self.with_neck:
            padding_constraints.update(self.neck.padding_constraints)
        images = ImageList.from_tensors(images, padding_constraints)
        return images

    def export_onnx(self, tensors):
        raise NotImplementedError

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)
