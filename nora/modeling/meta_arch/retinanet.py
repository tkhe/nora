from typing import List
from typing import Optional

from nora.config import configurable
from nora.modeling.backbone import Backbone
from nora.modeling.dense_head import DenseHead
from nora.modeling.neck import Neck
from .one_stage_detector import OneStageDetector

__all__ = ["RetinaNet"]


class RetinaNet(OneStageDetector):
    """
    support RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Optional[Neck] = None,
        box_head: DenseHead,
        pixel_mean: List[float],
        pixel_std: List[float],
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            box_head=box_head,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
