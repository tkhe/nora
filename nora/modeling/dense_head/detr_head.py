from typing import Dict
from typing import List
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nora.layers import MLP
from nora.layers import get_activation
from nora.modeling.matcher import Matcher
from nora.structures import BoxMode
from nora.structures import Boxes
from nora.structures import Instances
from nora.utils.comm import reduce_mean
from .base import DenseHead

__all__ = ["DETRHead"]


class DETRHead(DenseHead):
    def __init__(
        self,
        *,
        num_classes: int,
        embed_dim: int = 256,
        activation: Union[str, Dict] = "ReLU",
        matcher: Matcher,
        loss_cls: nn.Module,
        loss_iou: nn.Module,
        loss_reg: nn.Module,
        use_auxiliary_loss: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_auxiliary_loss = use_auxiliary_loss

        self.fc_cls = nn.Linear(embed_dim, num_classes + 1)
        self.reg_ffn = MLP(
            in_features=embed_dim,
            hidden_features=embed_dim,
            out_features=embed_dim,
            activation=activation,
        )
        self.activation = get_activation(activation)
        self.fc_reg = nn.Linear(embed_dim, 4)

        self.matcher = matcher

        self.loss_cls = loss_cls
        self.loss_iou = loss_iou
        self.loss_reg = loss_reg

    def forward(self, hidden_state: torch.Tensor):
        cls_scores = self.fc_cls(hidden_state)
        bbox_preds = self.fc_reg(self.activation(self.reg_ffn(hidden_state))).sigmoid()

        return cls_scores, bbox_preds

    def loss(
        self,
        all_layers_cls_scores: torch.Tensor,
        all_layers_bbox_preds: torch.Tensor,
        gt_instances: List[Instances],
    ):
        losses = {}

        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]

        gt_labels, gt_boxes, pred_boxes, bbox_targets = self.get_ground_truth(cls_scores, bbox_preds, gt_instances)
        loss_cls, loss_iou, loss_reg = self.calculate_loss(cls_scores, bbox_preds, gt_labels, gt_boxes, pred_boxes, bbox_targets)
        losses["loss_cls"] = loss_cls
        losses["loss_iou"] = loss_iou
        losses["loss_reg"] = loss_reg

        if self.use_auxiliary_loss:
            for idx, (cls_scores, bbox_preds) in enumerate(zip(all_layers_cls_scores[:-1], all_layers_bbox_preds[:-1])):
                gt_labels, gt_boxes, pred_boxes, bbox_targets = self.get_ground_truth(cls_scores, bbox_preds, gt_instances)
                loss_cls, loss_iou, loss_reg = self.calculate_loss(cls_scores, bbox_preds, gt_labels, gt_boxes, pred_boxes, bbox_targets)
                losses[f"aux.{idx}.loss_cls"] = loss_cls
                losses[f"aux.{idx}.loss_iou"] = loss_iou
                losses[f"aux.{idx}.loss_reg"] = loss_reg

        return losses

    def calculate_loss(self, cls_scores, bbox_preds, gt_labels, gt_boxes, pred_boxes, bbox_targets):
        gt_labels = torch.stack(gt_labels)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)

        num_pos = pos_mask.sum().item()

        num_pos = cls_scores.new_tensor([num_pos])
        num_pos = torch.clamp(reduce_mean(num_pos), min=1).item()

        loss_cls = self.loss_cls(cls_scores[valid_mask], gt_labels[valid_mask])

        gt_boxes = torch.stack(gt_boxes)
        pred_boxes = torch.stack(pred_boxes)
        bbox_targets = torch.stack(bbox_targets)

        loss_iou = self.loss_iou(pred_boxes[pos_mask], gt_boxes[pos_mask]) / num_pos
        loss_reg = self.loss_reg(bbox_preds[pos_mask], bbox_targets[pos_mask]) / num_pos

        return loss_cls, loss_iou, loss_reg

    @torch.no_grad()
    def get_ground_truth(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_instances: List[Instances],
    ):
        gt_labels = []
        matched_gt_boxes = []
        pred_boxes = []
        bbox_targets = []
        for cls_score, bbox_pred, gt_per_image in zip(cls_scores, bbox_preds, gt_instances):
            h, w = gt_per_image.image_size
            scale = bbox_pred.new_tensor([w, h, w, h]).unsqueeze(0)
            bbox_pred = BoxMode.convert(bbox_pred, BoxMode.CXCYWH_ABS, BoxMode.XYXY_ABS)
            bbox_pred = bbox_pred * scale

            pred_instances = Instances((h, w))
            pred_instances.pred_scores = cls_score
            pred_instances.pred_boxes = Boxes(bbox_pred)
            matched_idxs, matched_labels = self.matcher.match(pred_instances, gt_per_image)

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                gt_labels_i[matched_labels == 0] = self.num_classes
            else:
                matched_gt_boxes_i = torch.zeros_like(bbox_pred)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            pred_boxes.append(bbox_pred)
            bbox_targets.append(BoxMode.convert(matched_gt_boxes_i / scale, BoxMode.XYXY_ABS, BoxMode.CXCYWH_ABS))

        return gt_labels, matched_gt_boxes, pred_boxes, bbox_targets

    def inference(self, cls_scores, bbox_preds, image_sizes):
        results = []

        cls_scores = cls_scores[-1]
        bbox_preds = bbox_preds[-1]

        for image_idx, image_size in enumerate(image_sizes):
            cls_score = cls_scores[image_idx]
            bbox_pred = bbox_preds[image_idx]
            result = self.inference_single_image(cls_score, bbox_pred, image_size)
            results.append(result)
        return results

    def inference_single_image(self, cls_score, bbox_pred, image_size):
        scores, class_idxs = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)

        bbox_pred = BoxMode.convert(bbox_pred, BoxMode.CXCYWH_ABS, BoxMode.XYXY_ABS)
        bbox_pred[:, 0::2] *= image_size[1]
        bbox_pred[:, 1::2] *= image_size[0]

        result = Instances(image_size)
        result.pred_boxes = Boxes(bbox_pred)
        result.scores = scores
        result.pred_classes = class_idxs

        return result
