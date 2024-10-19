import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DistributedFocalLoss",
    "QualityFocalLoss",
]


def quality_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    scores: torch.Tensor,
    beta: float = 2.0,
    reduction: str = "mean",
):
    inputs_sigmoid = inputs.sigmoid()
    scale_factor = inputs_sigmoid
    zerolabel = scale_factor.new_zeros(inputs.shape)
    loss = F.binary_cross_entropy_with_logits(inputs, zerolabel, reduction="none") * scale_factor.pow(beta)

    bg_class_ind = inputs.size(1)
    pos = ((targets >= 0) & (targets < bg_class_ind)).nonzero().squeeze(1)
    pos_targets = targets[pos].long()
    scale_factor = scores[pos] - inputs_sigmoid[pos, pos_targets]
    loss[pos, pos_targets] = F.binary_cross_entropy_with_logits(
        inputs[pos, pos_targets],
        scores[pos],
        reduction="none",
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def distributed_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"):
    dis_left = targets.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - targets
    weight_right = targets - dis_left.float()
    loss = F.cross_entropy(inputs, dis_left, reduction='none') * weight_left + F.cross_entropy(inputs, dis_right, reduction='none') * weight_right

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class QualityFocalLoss(nn.Module):
    def __init__(
        self,
        beta: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets, scores):
        return self.loss_weight * quality_focal_loss(
            inputs,
            targets,
            scores,
            self.beta,
            self.reduction
        )


class DistributedFocalLoss(nn.Module):
    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        return self.loss_weight * distributed_focal_loss(inputs, targets)
