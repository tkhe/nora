from typing import Dict
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nora.modeling.backbone import Backbone
from nora.modeling.dense_head import DenseHead
from nora.modeling.neck import Neck
from nora.modeling.postprocessing import detector_postprocess
from .base_detector import Detector

__all__ = ["DETR"]


class DETR(Detector):
    """
    support DETR (https://arxiv.org/abs/2005.12872).
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Optional[Neck] = None,
        position_embedding: nn.Module,
        transformer: nn.Module,
        box_head: DenseHead,
        num_queries: int,
        pixel_mean: List[float],
        pixel_std: List[float],
    ):
        super().__init__()

        assert isinstance(backbone, Backbone)
        assert isinstance(box_head, DenseHead)

        if neck is not None:
            assert isinstance(neck, Neck)

        self.num_queries = num_queries

        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.box_head = box_head

        self.query_embed = nn.Embedding(num_queries, self.transformer.embed_dim)

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        if self.with_neck:
            features = self.neck(features)

        assert len(features) == 1, f"DETR only use the last level feature map, got {len(features)}"

        if self.training:
            N, _, H, W = images.tensor.size()
            image_masks = images.tensor.new_ones(N, H, W)
            for image_id in range(N):
                h, w = batched_inputs[image_id]["instances"].image_size
                image_masks[image_id, :h, :w] = 0
        else:
            N, _, H, W = images.tensor.size()
            image_masks = images.tensor.new_zeros(N, H, W)

        image_masks = F.interpolate(image_masks[None], size=features[self.transformer.in_features[-1]].shape[-2:]).to(torch.bool)[0]

        position_embedding = self.position_embedding(image_masks)

        hidden_states, _ = self.transformer(features, image_masks, self.query_embed.weight, position_embedding)

        predictions = self.box_head(hidden_states)

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
