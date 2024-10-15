# Copyright (c) Facebook, Inc. and its affiliates.

import torch

__all__ = ["pairwise_iou_rotated"]


def pairwise_iou_rotated(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    Arguments:
        boxes1 (Tensor[N, 5])
        boxes2 (Tensor[M, 5])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    return torch.ops.nora.box_iou_rotated(boxes1, boxes2)
