import torch

from .base import BoxCoder

__all__ = ["DistancePointBoxCoder"]


class DistancePointBoxCoder(BoxCoder):
    def get_deltas(self, points: torch.Tensor, target_boxes: torch.Tensor, strides: torch.Tensor):
        """
        Args:
            points (Tensor): of shape (N, 2)
            target_boxes (Tensor): of shape (N, )
        """
        l = (points[:, 0] - target_boxes[:, 0]) / strides[:, 0]
        t = (points[:, 1] - target_boxes[:, 1]) / strides[:, 1]
        r = (target_boxes[:, 2] - points[:, 0]) / strides[:, 0]
        b = (target_boxes[:, 3] - points[:, 1]) / strides[:, 1]
        return torch.stack([l, t, r, b], dim=-1)

    def apply_deltas(self, deltas: torch.Tensor, points: torch.Tensor, strides: torch.Tensor):
        x1 = points[:, 0] - deltas[:, 0] * strides[:, 0]
        y1 = points[:, 1] - deltas[:, 1] * strides[:, 1]
        x2 = points[:, 0] + deltas[:, 2] * strides[:, 0]
        y2 = points[:, 1] + deltas[:, 3] * strides[:, 1]
        return torch.stack([x1, y1, x2, y2], dim=-1)
