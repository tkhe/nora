import math
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from nora.layers import Conv2d
from nora.layers import ShapeSpec
from nora.layers import batched_nms
from nora.layers import cat
from nora.layers import nonzero_tuple
from nora.layers import permute_to_N_HWA_K
from nora.layers import weight_init
from nora.modeling.anchor_generator import PointGenerator
from nora.modeling.box_coder import BoxCoder
from nora.modeling.matcher import Matcher
from nora.structures import Boxes
from nora.structures import Instances
from nora.utils.comm import reduce_mean
from .base import DenseHead

__all__ = ["RTMDetHead"]


class RTMDetHead(DenseHead):
    """
    support RTMDetHead (https://arxiv.org/abs/2212.07784).
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        in_features: List[str],
        *,
        num_classes: int,
        widen_factor: float = 1.0,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        share_conv: bool = True,
        pred_kernel_size: int = 1,
        norm: Union[str, Dict] = "BN",
        activation: Union[str, Dict] = "SiLU",
        point_generator: PointGenerator,
        box_coder: BoxCoder,
        matcher: Matcher,
        loss_cls: nn.Module,
        loss_reg: nn.Module,
        test_score_threshold: float = 0.05,
        test_topk_candidates: int = 1000,
        test_nms_threshold: float = 0.5,
        max_detections_per_image: int = 100,
        prior_prob: float = 0.01,
    ):
        super().__init__()

        input_shape = {
            k: ShapeSpec(channels=int(v.channels * widen_factor), stride=v.stride)
            for k, v in input_shape.items()
        }
        feat_channels = int(feat_channels * widen_factor)

        assert len(set([input_shape[f].channels for f in in_features])) == 1, "Using different channels of input between levels is not currently supported!"

        in_channels = int(input_shape[in_features[0]].channels)

        self.strides = [input_shape[f].stride for f in in_features]
        self.in_features = in_features
        self.num_classes = num_classes

        self.test_score_threshold = test_score_threshold
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_threshold = test_nms_threshold
        self.max_detections_per_image = max_detections_per_image

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()

        for n in range(len(in_features)):
            cls_convs = []
            reg_convs = []
            for i in range(stacked_convs):
                channels = in_channels if i == 0 else feat_channels
                cls_convs.append(Conv2d(channels, feat_channels, kernel_size=3, stride=1, norm=norm, activation=activation))
                reg_convs.append(Conv2d(channels, feat_channels, kernel_size=3, stride=1, norm=norm, activation=activation))
            self.cls_convs.append(nn.Sequential(*cls_convs))
            self.reg_convs.append(nn.Sequential(*reg_convs))

            self.rtm_cls.append(Conv2d(feat_channels, num_classes, kernel_size=pred_kernel_size, stride=1))
            self.rtm_reg.append(Conv2d(feat_channels, 4, kernel_size=pred_kernel_size, stride=1))

        self.point_generator = point_generator
        self.box_coder = box_coder
        self.matcher = matcher

        self.loss_cls = loss_cls
        self.loss_reg = loss_reg

        if share_conv:
            for n in range(len(in_features)):
                for i in range(stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2d)):
                weight_init.normal_fill(m, mean=0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                weight_init.constant_fill(m, val=1)

        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        for rtm_cls in self.rtm_cls:
            weight_init.bias_fill(rtm_cls, bias_value)

    def forward(self, features: Dict[str, torch.Tensor]):
        features = [features[f] for f in self.in_features]

        logits = []
        bbox_reg = []
        for idx, x in enumerate(features):
            logits.append(self.rtm_cls[idx]((self.cls_convs[idx](x))))
            bbox_reg.append(self.rtm_reg[idx]((self.reg_convs[idx](x))))

        return logits, bbox_reg

    def loss(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        gt_instances: List[Instances],
    ):
        points = self.point_generator(cls_scores)
        strides = [p.new_full(p.size(), s) for p, s in zip(points, self.strides)]

        cls_scores = torch.cat([permute_to_N_HWA_K(x, self.num_classes) for x in cls_scores], dim=1)
        bbox_preds = torch.cat([permute_to_N_HWA_K(x, 4) for x in bbox_preds], dim=1)

        strides = torch.cat(strides, dim=0)
        points = torch.cat(points, dim=0)

        pred_boxes = torch.stack([self.box_coder.apply_deltas(k, points, strides) for k in bbox_preds], dim=0)
        gt_labels, gt_boxes, match_metrics = self.get_ground_truth(
            points,
            cls_scores,
            pred_boxes,
            strides,
            gt_instances,
        )

        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)
        match_metrics = torch.stack(match_metrics)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_samples = pos_mask.sum().item()

        avg_factor = reduce_mean(match_metrics.sum()).clamp_(min=1).item()

        loss_cls = self.loss_cls(cls_scores[valid_mask], gt_labels[valid_mask], match_metrics[valid_mask]) / avg_factor
        if num_pos_samples > 0:
            loss_reg = (self.loss_reg(pred_boxes[pos_mask], torch.stack(gt_boxes)[pos_mask]) * match_metrics[pos_mask]).sum() / avg_factor
        else:
            loss_reg = bbox_preds.sum() * 0.0

        return {
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
        }

    @torch.no_grad()
    def get_ground_truth(
        self,
        points,
        pred_logits,
        pred_boxes,
        strides,
        gt_instances,
    ):
        gt_labels = []
        matched_gt_boxes = []
        matched_metrics = []
        for idx, gt_per_image in enumerate(gt_instances):
            matched_idxs, matched_labels, matched_metrics_i = self.matcher.match(
                pred_logits[idx],
                pred_boxes[idx],
                points,
                gt_per_image.gt_boxes,
                gt_per_image.gt_classes,
                strides,
            )

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                gt_labels_i[matched_labels == 0] = self.num_classes
                gt_labels_i[matched_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(pred_boxes)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_metrics.append(matched_metrics_i)

        return gt_labels, matched_gt_boxes, matched_metrics

    def inference(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        points = self.point_generator(cls_scores)
        strides = [p.new_full(p.size(), s) for p, s in zip(points, self.strides)]

        cls_scores = [permute_to_N_HWA_K(x, self.num_classes) for x in cls_scores]
        bbox_preds = [permute_to_N_HWA_K(x, 4) for x in bbox_preds]

        results = []
        for image_idx, image_size in enumerate(image_sizes):
            cls_scores_per_image = [x[image_idx] for x in cls_scores]
            bbox_preds_per_image = [x[image_idx] for x in bbox_preds]
            results_per_image = self.inference_single_image(
                cls_scores_per_image,
                bbox_preds_per_image,
                points,
                strides,
                image_size,
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        points: List[torch.Tensor],
        strides: List[torch.Tensor],
        image_size: Tuple[int, int],
    ):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, points_i, strides_i in zip(cls_scores, bbox_preds, points, strides):
            predicted_prob = box_cls_i.flatten().sigmoid_()

            keep_idxs = predicted_prob > self.test_score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            point_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[point_idxs]
            points_i = points_i[point_idxs]
            strides_i = strides_i[point_idxs]
            predicted_boxes = self.box_coder.apply_deltas(box_reg_i, points_i, strides_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all, scores_all, class_idxs_all]]

        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result
