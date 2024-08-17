# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import math
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nora.layers import Conv2d
from nora.layers import ShapeSpec
from nora.layers import batched_nms
from nora.layers import cat
from nora.layers import get_norm
from nora.layers import nonzero_tuple
from nora.layers import permute_to_N_HWA_K
from nora.layers import weight_init
from nora.modeling.anchor_generator import AnchorGenerator
from nora.modeling.box_coder import BoxCoder
from nora.modeling.matcher import Matcher
from nora.structures import Boxes
from nora.structures import Instances
from nora.structures import pairwise_iou
from nora.utils.events import get_event_storage
from .base import DenseHead

__all__ = ["RetinaNetHead"]

logger = logging.getLogger(__name__)


class RetinaNetHead(DenseHead):
    """
    support RetinaNetHead proposed in https://arxiv.org/abs/1708.02002.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        in_features: List[str],
        *,
        num_classes: int,
        conv_dims: List[int],
        norm: Union[str, Dict] = "",
        activation: Union[str, Dict] = "ReLU",
        anchor_generator: AnchorGenerator,
        box_coder: BoxCoder,
        matcher: Matcher,
        loss_cls: nn.Module,
        loss_reg: nn.Module,
        reg_decoded_box: bool = False,
        test_score_threshold: float = 0.05,
        test_topk_candidates: int = 1000,
        test_nms_threshold: float = 0.5,
        max_detections_per_image: int = 100,
        prior_prob: float = 0.01,
    ):
        """
        Args:
            input_shape (Dict[str, ShapeSpec]): input shape
            in_features (List[str]): input feature names
            num_classes (int): number of classes.
            conv_dims (List[int]): dimensions for each convolutional layer
            norm (str or Dict[str, str]): normalization for conv layers except for the two output layers.
            activation (str or Dict[str, str]): activation for conv layers except for the two output layers.
            anchor_generator (AnchorGenerator): anchor generator creates anchors from a list of features.
            box_coder (BoxCoder): defines the transformation from anchors boxes to instance boxes.
            matcher (Matcher): label the anchors by matching them with ground truth.
            loss_cls (nn.Module): classification loss.
            loss_reg (nn.Module): regression loss.
            reg_decoded_box (bool): whether to use decoded box for computing loss.
            test_score_threshold (float): inference cls score threshold.
                only anchors with score > test_score_threshold are considered for inference.
            test_topk_candidates (int): select topk candidates before NMS.
            test_nms_threshold (float): overlap threshold used for NMS.
            max_detections_per_image (int): maximum number of detections per image during inference.
            prior_prob (float): prior weight for computing bias
        """
        super().__init__()

        assert len(set([input_shape[f].channels for f in in_features])) == 1, "Using different channels of input between levels is not currently supported!"

        num_anchors = anchor_generator.num_anchors
        assert (len(set(num_anchors)) == 1), "Using different number of anchors between levels is not currently supported!"

        num_anchors = num_anchors[0]

        norm_class = type(get_norm(norm, input_shape[in_features[0]].channels))
        if norm_class in (nn.BatchNorm2d, nn.SyncBatchNorm):
            logger.warning(f"Shared BatchNorm (type={str(norm_class)}) does not work well for BN, SyncBN, expect poor results.")

        self.in_features = in_features
        self.num_classes = num_classes
        self.test_score_threshold = test_score_threshold
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_threshold = test_nms_threshold
        self.max_detections_per_image = max_detections_per_image

        cls_subnet = []
        bbox_subnet = []
        for in_channels, out_channels in zip([input_shape[in_features[0]].channels] + list(conv_dims[:-1]), conv_dims):
            cls_subnet.append(Conv2d(in_channels, out_channels, kernel_size=3, stride=1, norm=norm, activation=activation))
            bbox_subnet.append(Conv2d(in_channels, out_channels, kernel_size=3, stride=1, norm=norm, activation=activation))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = Conv2d(conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1)
        self.bbox_pred = Conv2d(conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1)

        self.anchor_generator = anchor_generator
        self.box_coder = box_coder
        self.matcher = matcher

        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.reg_decoded_box = reg_decoded_box

        self.loss_normalizer = 100
        self.loss_normalizer_momentum = 0.9

        # initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    weight_init.normal_fill(layer, mean=0, std=0.01)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        weight_init.bias_fill(self.cls_score, bias_value)

    def forward(self, features: Dict[str, torch.Tensor]):
        """
        Args:
            features (Dict[str, torch.Tensor]): FPN feature map tensors.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.

        """
        features = [features[f] for f in self.in_features]

        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg

    def loss(self, pred_logits, pred_anchor_deltas, gt_instances):
        anchors = self.anchor_generator(pred_logits)

        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        gt_labels, gt_boxes = self.get_ground_truth(anchors, gt_instances)

        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()

        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)

        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (1 - self.loss_normalizer_momentum) * max(num_pos_anchors, 1)

        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[:, :-1]  # no loss for the last (background) class
        loss_cls = self.loss_cls(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
        )

        anchors = Boxes.cat(anchors).tensor
        if self.reg_decoded_box:
            pred_boxes = [self.box_coder.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)]
            loss_reg = self.loss_reg(
                torch.stack(pred_boxes)[pos_mask],
                torch.stack(gt_boxes)[pos_mask],
            )
        else:
            gt_anchor_deltas = [self.box_coder.get_deltas(anchors, k) for k in gt_boxes]
            gt_anchor_deltas = torch.stack(gt_anchor_deltas)
            loss_reg = self.loss_reg(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
            )
        return {
            "loss_cls": loss_cls / self.loss_normalizer,
            "loss_reg": loss_reg / self.loss_normalizer,
        }

    @torch.no_grad()
    def get_ground_truth(self, anchors: Boxes, gt_instances: List[Instances]):
        anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.matcher.match(match_quality_matrix)

            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def inference(self, pred_logits, pred_anchor_deltas, image_sizes):
        anchors = self.anchor_generator(pred_logits)

        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        results = []

        for image_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[image_idx] for x in pred_logits]
            deltas_per_image = [x[image_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(anchors, pred_logits_per_image, deltas_per_image, image_size)
            results.append(results_per_image)
        return results

    def inference_single_image(
        self,
        anchors: List[Boxes],
        box_cls: List[torch.Tensor],
        box_delta: List[torch.Tensor],
        image_size: Tuple[int, int],
     ):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box_coder.apply_deltas(box_reg_i, anchors_i.tensor)

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
