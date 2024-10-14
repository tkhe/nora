# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nora.utils.registry import Registry

__all__ = [
    "FrozenBatchNorm2d",
    "LayerNorm2d",
    "get_norm",
]

NORM_REGISTRY = Registry("Norm")


@NORM_REGISTRY.register("FrozenBN")
class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)
        self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def __repr__(self):
        return f"FrozenBatchNorm2d(num_features={self.num_features}, eps={self.eps})"

    @classmethod
    def convert_frozen_batch_norm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batch_norm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    @classmethod
    def convert_frozenbatchnorm2d_to_batchnorm2d(cls, module: nn.Module) -> nn.Module:
        """
        Convert all FrozenBatchNorm2d to BatchNorm2d

        Args:
            module (torch.nn.Module):

        Returns:
            If module is FrozenBatchNorm2d, returns a new module.
            Otherwise, in-place convert module and return it.
        """
        res = module
        if isinstance(module, FrozenBatchNorm2d):
            res = nn.BatchNorm2d(module.num_features, module.eps)

            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data.clone().detach()
            res.running_var.data = module.running_var.data.clone().detach()
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozenbatchnorm2d_to_batchnorm2d(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


@NORM_REGISTRY.register("LN2d")
class LayerNorm2d(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, channel_last: bool = False):
        if channel_last:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


NORM_REGISTRY.register("BN", nn.BatchNorm2d)
NORM_REGISTRY.register("GN", nn.GroupNorm)
NORM_REGISTRY.register("SyncBN", nn.SyncBatchNorm)
NORM_REGISTRY.register("LN", nn.LayerNorm)


def get_norm(norm: Union[str, Dict], out_channels) -> Optional[nn.Module]:
    """
    Args:
        norm (str | dict)
        out_channels

    Returns:
        nn.Module or None
    """
    if isinstance(norm, str):
        norm = {"_target_": norm}

    if norm is None or norm.get("_target_", "") == "":
        return None

    if norm.get("_target_") == "GN":
        kwargs = {"num_channels": out_channels}
        if "num_groups" not in norm:
            kwargs["num_groups"] = 32
        return NORM_REGISTRY.build(norm, **kwargs)
    elif norm.get("_target_") == "LN" or norm.get("_target_") == "LN2d":
        return NORM_REGISTRY.build(norm, normalized_shape=out_channels)
    else:
        return NORM_REGISTRY.build(norm, num_features=out_channels)
