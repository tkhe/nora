from typing import Dict
from typing import List
from typing import Optional

import torch

from nora.config import instantiate
from nora.modeling.backbone import Backbone
from nora.modeling.neck import Neck
from nora.modeling.postprocessing import sem_seg_postprocess
from nora.modeling.sem_seg_head import SemSegHead
from nora.structures import ImageList
from .base import BaseModel

__all__ = ["SemanticSegmentor"]


class SemanticSegmentor(BaseModel):
    """
    Base class for semantic segmentor.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Optional[Neck] = None,
        sem_seg_head: SemSegHead,
        pixel_mean: List[float],
        pixel_std: List[float],
    ):
        super().__init__()

        assert isinstance(backbone, Backbone)
        assert isinstance(sem_seg_head, SemSegHead)

        if neck is not None:
            assert isinstance(neck, Neck)

        self.backbone = backbone
        self.neck = neck
        self.sem_seg_head = sem_seg_head

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def with_sem_seg_head(self) -> bool:
        return hasattr(self, "sem_seg_head") and self.sem_seg_head is not None

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        if self.with_neck:
            features = self.neck(features)
        predictions = self.sem_seg_head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "sem_seg" in batched_inputs[0], "semantic segmentation annotations are missing in training!"

            targets = [self._move_to_current_device(x["sem_seg"]) for x in batched_inputs]
            padding_constraints = {}
            padding_constraints.update(self.backbone.padding_constraints)
            if self.with_neck:
                padding_constraints.update(self.neck.padding_constraints)
            targets = ImageList.from_tensors(
                targets,
                padding_constraints,
                pad_value=self.sem_seg_head.ignore_value,
            ).tensor

            losses = self.sem_seg_head.loss(predictions, targets)
            return losses
        else:
            results = self.sem_seg_head.inference(predictions, images.tensor.shape[-2:])

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(results_per_image, image_size, height, width)
                processed_results.append({"sem_seg": r})
            return processed_results

    def export_onnx(self, tensors):
        tensors = self.backbone(tensors)
        if self.with_neck:
            tensors = self.neck(tensors)
        tensors = self.sem_seg_head(tensors)
        return tensors
