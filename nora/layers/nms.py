# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torchvision.ops import boxes as box_ops

__all__ = [
    "batched_nms",
]


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float):
    """
    Same as torchvision.ops.boxes.batched_nms, but with float().
    """
    assert boxes.shape[-1] == 4

    # Note: Torchvision already has a strategy (https://github.com/pytorch/vision/issues/1311)
    # to decide whether to use coordinate trick or for loop to implement batched_nms. So we
    # just call it directly.
    # Fp16 does not have enough range for batched NMS, so adding float().
    return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)
