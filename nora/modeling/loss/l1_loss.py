# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BalancedL1Loss",
    "L1Loss",
    "SmoothL1Loss",
]


def smooth_l1_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
    ::
                      | 0.5 * x ** 2 / beta   if abs(x) < beta
        smoothl1(x) = |
                      | abs(x) - 0.5 * beta   otherwise,

    where x = inputs - targets.

    Smooth L1 loss is related to Huber loss, which is defined as:
    ::
                    | 0.5 * x ** 2                  if abs(x) < beta
         huber(x) = |
                    | beta * (abs(x) - 0.5 * beta)  otherwise

    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:

     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.

    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.

    Args:
        inputs (Tensor): input tensor of any shape
        targets (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'sum': The output will be summed.

    Returns:
        The loss with the reduction option applied.

    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(inputs - targets)
    else:
        n = torch.abs(inputs - targets)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def balanced_l1_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 1.0,
    gamma: float = 1.5,
    reduction: str = "none",
):
    """
    Args:
        inputs (Tensor): input tensor with shape (N, 4).
        targets (Tensor): target tensor with shape (N, 4).
        alpha (float): alpha parameter in balanced L1 loss.
        beta (float): beta parameter in balanced L1 loss.
        gamma (float): gamma parameter in balanced L1 loss.
        reduction: 'none' | 'mean' | 'sum'
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'sum': The output will be summed.

    Returns:
        The loss with the reduction option applied.
    """
    assert beta > 0

    if targets.numel() == 0 and reduction == "mean":
        return inputs.sum() * 0.0
    
    assert inputs.size() == targets.size()

    diff = torch.abs(inputs - targets)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta
    )

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class L1Loss(nn.Module):
    """
    A wrapper for nn.L1Loss.
    """

    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return self.loss_weight * F.l1_loss(inputs, targets, reduction=self.reduction)


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss defined in Fast R-CNN (https://arxiv.org/abs/1504.08083).
    """

    def __init__(
        self,
        beta: float,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return self.loss_weight * smooth_l1_loss(inputs, targets, self.beta, self.reduction)


class BalancedL1Loss(nn.Module):
    """
    Balanced L1 Loss defined in Libra R-CNN (https://arxiv.org/abs/1904.02701).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 1.5,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return self.loss_weight * balanced_l1_loss(
            inputs,
            targets,
            self.alpha,
            self.beta,
            self.gamma,
            self.reduction,
        )
