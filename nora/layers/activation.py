from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nora.utils.registry import Registry

__all__ = [
    "ACTIVATION_REGISTRY",
    "Activation",
    "HardMish",
    "HardSigmoid",
    "HardSwish",
    "LeakyReLU",
    "Mish",
    "ReLU",
    "ReLU6",
    "SiLU",
    "Sigmoid",
    "get_activation",
]

ACTIVATION_REGISTRY = Registry("Activation")


class Activation(nn.Module, ABC):
    """
    Abstract base class for activation.
    """

    def __init__(self, inplace: bool = True):
        """
        Args:
            inplace (bool): can optionally do the operation in-place. Default: `True`
        """
        super().__init__()

        self.inplace = inplace

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


@ACTIVATION_REGISTRY.register()
class ReLU(Activation):
    """
    Same as `torch.nn.ReLU`, but `inplace` is `True` by default.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x, self.inplace)


@ACTIVATION_REGISTRY.register()
class ReLU6(Activation):
    """
    Same as `torch.nn.ReLU6`, but `inplace` is `True` by default.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x, self.inplace)


@ACTIVATION_REGISTRY.register()
class Sigmoid(Activation):
    """
    Same as `torch.nn.Sigmoid`, but support in-place operation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sigmoid_() if self.inplace else x.sigmoid()


@ACTIVATION_REGISTRY.register()
class SiLU(Activation):
    """
    Same as `torch.nn.SiLU`, but `inplace` is `True` by default.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x, self.inplace)


@ACTIVATION_REGISTRY.register()
class HardSigmoid(Activation):
    """
    Same as `torch.nn.Hardsigmoid`, but `inplace` is `True` by default.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardsigmoid(x, self.inplace)


@ACTIVATION_REGISTRY.register()
class HardSwish(Activation):
    """
    Same as `torch.nn.Hardswish`, but `inplace` is `True` by default.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardswish(x, self.inplace)


@ACTIVATION_REGISTRY.register()
class Mish(Activation):
    """
    Same as `torch.nn.Mish`, but `inplace` is `True` by default.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x, self.inplace)


@ACTIVATION_REGISTRY.register()
class HardMish(Activation):
    """
    Modified from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/activations.py
    """

    def __init__(self, inplace: bool = True):
        super().__init__(inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
        else:
            return 0.5 * x * (x + 2).clamp(min=0, max=2)


@ACTIVATION_REGISTRY.register()
class LeakyReLU(Activation):
    """
    Same as `torch.nn.LeakyReLU`, but `negative_slope` is `0.1` and `inplace` is `True` by default.
    """

    def __init__(self, negative_slope: float = 0.1, inplace: bool = True):
        super().__init__(inplace)

        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"negative_slope={self.negative_slope}{inplace_str}"


ACTIVATION_REGISTRY.register("GELU", nn.GELU)


def get_activation(activation: Union[str, Dict]) -> Optional[Activation]:
    """
    Args:
        activation (str | dict)

    Returns:
        nn.Module or None: the activation layer.
    """
    if isinstance(activation, str):
        activation = {"_target_": activation}

    if activation is None or activation.get("_target_", "") == "":
        return None

    return ACTIVATION_REGISTRY.build(activation)
