from typing import Dict
from typing import List
from typing import Optional

import torch

from nora.modeling.backbone import Backbone
from nora.modeling.dense_head import DenseHead
from nora.modeling.neck import Neck
from nora.modeling.postprocessing import detector_postprocess
from .base_detector import Detector

__all__ = ["OneStageDetector"]


class OneStageDetector(Detector):
    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Optional[Neck] = None,
        box_head: DenseHead,
        pixel_mean: List[float],
        pixel_std: List[float],
    ):
        super().__init__()

        assert isinstance(backbone, Backbone)
        assert isinstance(box_head, DenseHead)

        if neck is not None:
            assert isinstance(neck, Neck)

        self.backbone = backbone
        self.neck = neck
        self.box_head = box_head

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        if self.with_neck:
            features = self.neck(features)
        predictions = self.box_head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            losses = self.box_head.loss(*predictions, gt_instances)
            return losses
        else:
            results = self.box_head.inference(*predictions, images.image_sizes)
            if torch.jit.is_scripting():
                return results

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def export_onnx(self, tensors):
        tensors = self.backbone(tensors)
        if self.with_neck:
            tensors = self.neck(tensors)
        tensors = self.box_head(tensors)
        return tensors
