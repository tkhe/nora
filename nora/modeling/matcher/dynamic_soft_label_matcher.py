import torch
import torch.nn.functional as F

from nora.structures import Boxes
from nora.structures import pairwise_iou
from nora.structures import pairwise_point_box_distance
from .base import Matcher

__all__ = ["DynamicSoftLabelMatcher"]


class DynamicSoftLabelMatcher(Matcher):
    EPS = 1e-7
    INF = 1e8

    def __init__(
        self,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
    ):
        super().__init__()

        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight

    def match(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        priors: torch.Tensor,
        gt_boxes: Boxes,
        gt_classes: torch.Tensor,
        strides: torch.Tensor,
    ):
        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)

        if num_gt == 0 or num_pred == 0:
            default_matches = gt_classes.new_full((num_pred,), 0, dtype=torch.int64)
            default_match_labels = gt_classes.new_full((num_pred,), 0, dtype=torch.int8)
            default_match_metrics = gt_classes.new_full((num_pred,), 0, dtype=torch.float32)
            return default_matches, default_match_labels, default_match_metrics

        prior_dim = priors.size(1)
        if prior_dim == 2:
            prior_center = priors
        elif prior_dim == 4:
            prior_center = (priors[:, [0, 2]] + priors[:, [1, 3]]) / 2
        else:
            raise NotImplementedError

        # (N_points, N_boxes)
        is_in_gts = self.find_inside_points(gt_boxes, prior_center)
        # (N_points,)
        valid_mask = is_in_gts.sum(dim=-1) > 0

        valid_pred_boxes = pred_boxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_pred_boxes.size(0)

        if num_valid == 0:
            default_matches = gt_classes.new_full((num_pred,), 0, dtype=torch.int64)
            default_match_labels = gt_classes.new_full((num_pred,), 0, dtype=torch.int8)
            default_match_metrics = gt_classes.new_full((num_pred,), 0, dtype=torch.float32)
            return default_matches, default_match_labels, default_match_metrics

        gt_centers = gt_boxes.get_centers()
        valid_priors = priors[valid_mask]

        valid_strides = strides[valid_mask]

        distance = (valid_priors[:, None, :2] - gt_centers[None, :, :]).pow(2).sum(-1).sqrt() / valid_strides[:, 0, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        pairwise_ious = pairwise_iou(Boxes(valid_pred_boxes), gt_boxes)
        iou_cost = -torch.log(pairwise_ious + self.EPS) * self.iou_weight

        num_classes = pred_scores.size(-1)
        gt_onehot_label = F.one_hot(
            gt_classes.to(torch.int64),
            num_classes,
        ).float().unsqueeze(0).repeat(num_valid, 1, 1)
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        soft_cls_cost = F.binary_cross_entropy_with_logits(
            valid_pred_scores,
            soft_label,
            reduction="none",
        ) * scale_factor.abs().pow(2.0)
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix,
            pairwise_ious,
            num_gt,
            valid_mask,
        )

        match_labels = gt_classes.new_full((num_pred, ), 0, dtype=torch.int8)
        match_labels[valid_mask] = 1
        matches = gt_classes.new_full((num_pred,), 0, dtype=torch.int64)
        matches[valid_mask] = matched_gt_inds
        match_metrics = gt_classes.new_full((num_pred,), 0, dtype=torch.float32)
        match_metrics[valid_mask] = matched_pred_ious
        return matches, match_labels, match_metrics

    def find_inside_points(self, gt_boxes: Boxes, points: torch.Tensor):
        deltas = pairwise_point_box_distance(points, gt_boxes)
        is_in_gts = deltas.min(dim=-1).values > 0
        return is_in_gts

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
