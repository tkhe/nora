from typing import List
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CrossEntropyLoss"]


class CrossEntropyLoss(nn.Module):
    """
    A wrapper of `nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        if targets.numel() == 0 and self.reduction == "mean":
            return inputs.sum() * 0.0

        weight = inputs.new_tensor(self.weight) if self.weight is not None else None
        return self.loss_weight * F.cross_entropy(
            inputs,
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

    @property
    def name(self) -> str:
        return "loss_ce"
