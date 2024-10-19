from abc import ABC
from abc import abstractmethod

import torch

from nora.structures import BoxMode
from nora.structures import Instances
from nora.structures import pairwise_iou
from nora.structures.boxes import pairwise_intersection

__all__ = [
    "CrossEntropyCost",
    "FocalLossCost",
    "GIoUCost",
    "IoUCost",
    "L1Cost",
    "MatchCost",
]


class MatchCost(ABC):
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def __call__(
        self,
        pred_instances: Instances,
        gt_instances: Instances,
    ):
        raise NotImplementedError


class L1Cost(MatchCost):
    def __init__(self, box_format: str = "xywh", weight: float = 1.0):
        super().__init__(weight)

        assert box_format in {"xyxy", "xywh"}

        self.box_format = box_format

    def __call__(self, pred_instances: Instances, gt_instances: Instances):
        pred_boxes = pred_instances.pred_boxes.tensor
        gt_boxes = gt_instances.gt_boxes.tensor

        if self.box_format == "xywh":
            pred_boxes = BoxMode.convert(pred_boxes, BoxMode.XYXY_ABS, BoxMode.CXCYWH_ABS)
            gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYXY_ABS, BoxMode.CXCYWH_ABS)

        h, w = gt_instances.image_size

        scale = gt_boxes.new_tensor([w, h, w, h]).unsqueeze(0)
        gt_boxes = gt_boxes / scale
        pred_boxes = pred_boxes / scale

        cost = torch.cdist(pred_boxes, gt_boxes, p=1)
        return self.weight * cost


class IoUCost(MatchCost):
    def __init__(self, weight: float = 1):
        super().__init__(weight)

    def __call__(self, pred_instances: Instances, gt_instances: Instances):
        pred_boxes = pred_instances.pred_boxes
        gt_boxes = gt_instances.gt_boxes

        ious = pairwise_iou(pred_boxes, gt_boxes)
        # The 1 is a constant that doesn't change the matching, so omitted.
        cost = -ious
        return self.weight * cost


class GIoUCost(MatchCost):
    def __init__(self, weight: float = 1):
        super().__init__(weight)

    def __call__(self, pred_instances: Instances, gt_instances: Instances):
        pred_boxes = pred_instances.pred_boxes
        gt_boxes = gt_instances.gt_boxes

        inter = pairwise_intersection(pred_boxes, gt_boxes)
        area1 = pred_boxes.area()
        area2 = gt_boxes.area()
        union = area1[:, None] + area2 - inter
        ious = torch.where(
            inter > 0,
            inter / union,
            torch.zeros(1, dtype=inter.dtype, device=inter.device),
        )
        lt = torch.min(pred_boxes.tensor[:, None, :2], gt_boxes.tensor[:, :2])
        rb = torch.max(pred_boxes.tensor[:, None, 2:], gt_boxes.tensor[:, 2:])
        wh = (rb - lt).clamp(min=0)
        area = wh[:, :, 0] * wh[:, :, 1]
        gious = ious - (area - union) / (area + 1e-7)

        cost = -gious

        return self.weight * cost


class CrossEntropyCost(MatchCost):
    def __init__(self, weight: float = 1):
        super().__init__(weight)

    def __call__(self, pred_instances: Instances, gt_instances: Instances):
        pred_scores = pred_instances.pred_scores
        gt_classes = gt_instances.gt_classes

        pred_scores = pred_scores.softmax(-1)
        cost = -pred_scores[:, gt_classes]

        return self.weight * cost


class FocalLossCost(MatchCost):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: float = 1,
    ):
        super().__init__(weight)

        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, pred_instances: Instances, gt_instances: Instances):
        pred_scores = pred_instances.pred_scores.sigmoid()
        gt_classes = gt_instances.gt_classes

        neg_cost = (1 - self.alpha) * (pred_scores ** self.gamma) * (-(1 - pred_scores + 1e-8).log())
        pos_cost = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
        cost = pos_cost[:, gt_classes] - neg_cost[:, gt_classes]

        return self.weight * cost
