import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
]


class BCELoss(nn.Module):
    """
    A wrapper for `nn.BCELoss`.
    """

    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight


    def forward(self, inputs, targets):
        if targets.numel() == 0 and self.reduction == "mean":
            return inputs.sum() * 0.0

        return self.loss_weight * F.binary_cross_entropy(
            inputs,
            targets,
            reduction=self.reduction,
        )

    @property
    def name(self) -> str:
        return "loss_bce"


class BCEWithLogitsLoss(nn.Module):
    """
    A wrapper for `nn.BCEWithLogitsLoss`.
    """

    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight


    def forward(self, inputs, targets):
        if targets.numel() == 0 and self.reduction == "mean":
            return inputs.sum() * 0.0

        return self.loss_weight * F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction=self.reduction,
        )

    @property
    def name(self) -> str:
        return "loss_bce"
