# Copyright (c) Facebook, Inc. and its affiliates.

import functools
from typing import List
from typing import Optional

import torch

from nora.utils.env import TORCH_VERSION

__all__ = [
    "cat",
    "channel_shuffle",
    "inverse_sigmoid",
    "make_divisible",
    "move_device_like",
    "nonzero_tuple",
    "permute_to_N_HWA_K",
    "shapes_to_tensor",
]


def shapes_to_tensor(x: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.

    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)

    if torch.jit.is_tracing():
        assert all([isinstance(t, torch.Tensor) for t in x]), "Shape should be tensor during tracing!"

        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret

    return torch.as_tensor(x, device=device)


def check_if_dynamo_compiling():
    if TORCH_VERSION >= (2, 1):
        from torch._dynamo import is_compiling

        return is_compiling()
    else:
        return False


def disable_torch_compiler(func):
    if TORCH_VERSION >= (2, 1):
        # Use the torch.compiler.disable decorator if supported
        @torch.compiler.disable
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
    else:
        # Return the function unchanged if torch.compiler.disable is not supported
        return func


def cat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))

    if len(tensors) == 1:
        return tensors[0]

    return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)


@torch.jit.script_if_tracing
def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Tracing friendly way to cast tensor to another tensor's device. Device will be treated
    as constant during tracing, scripting the casting process as whole can workaround this issue.
    """
    return src.to(dst.device)


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += v

    return new_v


def permute_to_N_HWA_K(tensor: torch.Tensor, K: int) -> torch.Tensor:
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape

    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """
    N, C, H, W = x.size()
    assert C % groups == 0, f"'num_channels' should be divisible by 'groups', but got num_channels={C}, groups={groups}"

    channels_per_group = C // groups
    x = x.view(N, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(N, -1, H, W)
    return x


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3):
    """
    The inverse function for sigmoid activation function.

    NOTE:
        It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
