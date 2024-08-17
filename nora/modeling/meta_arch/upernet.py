from typing import List
from typing import Optional

from nora.config import configurable
from nora.modeling.backbone import Backbone
from nora.modeling.neck import Neck
from nora.modeling.sem_seg_head import SemSegHead
from .base_semantic_segmentor import SemanticSegmentor

__all__ = ["UPerNet"]


class UPerNet(SemanticSegmentor):
    """
    support UPerNet (https://arxiv.org/abs/1807.10221).
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Optional[Neck] = None,
        sem_seg_head: SemSegHead,
        pixel_mean: List[float],
        pixel_std: List[float],
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            sem_seg_head=sem_seg_head,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
