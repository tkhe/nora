# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import math
from typing import List
from typing import Union

import torch
import torch.nn as nn

from nora.layers import move_device_like
from nora.structures import Boxes

__all__ = [
    "AnchorGenerator",
    "PointGenerator",
]


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super().__init__()

        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(size: List[int], stride: int, offset: float, target_device_tensor: torch.Tensor):
    grid_height, grid_width = size
    shifts_x = move_device_like(torch.arange(offset * stride, grid_width * stride, step=stride, dtype=torch.float32),target_device_tensor)
    shifts_y = move_device_like(torch.arange(offset * stride, grid_height * stride, step=stride, dtype=torch.float32),target_device_tensor)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.

    Returns:
        list[list[float]]: param for each feature
    """
    assert isinstance(params, collections.abc.Sequence), f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"

    if not isinstance(params[0], collections.abc.Sequence):  # params is list[float]
        return [params] * num_features

    if len(params) == 1:
        return list(params) * num_features

    assert len(params) == num_features, f"Got {name} of length {len(params)} in anchor generator, but the number of input features is {num_features}!"

    return params


class AnchorGenerator(nn.Module):
    """
    Compute anchors in the standard ways described in `Faster R-CNN`.
    """

    def __init__(
        self,
        sizes: Union[List[List[float]], List[float]],
        aspect_ratios: Union[List[List[float]], List[float]],
        strides: List[int],
        offset: float = 0.5
    ):
        """
        Args:
            sizes (list[list[float]] or list[float]):
                If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        super().__init__()

        assert 0.0 <= offset < 1.0, f"`offset` should be in [0, 1), but got {offset}!"

        self.strides = strides
        self.num_features = len(self.strides)
        self.offset = offset

        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(s, a).float()
            for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    @property
    def num_anchors(self) -> List[int]:
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.

                In standard RPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        # buffers() not supported by torchscript. use named_buffers() instead
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors.

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_feature_maps]


class PointGenerator(nn.Module):
    def __init__(self, strides: List[int], offset: float = 0.0):
        super().__init__()

        assert 0.0 <= offset < 1.0, f"`offset` should be in [0, 1), but got {offset}!"

        self.strides = strides
        self.offset = offset

        self.cell_points = self._calculate_points()

    def _calculate_points(self):
        cell_points = [torch.tensor([0.0, 0.0]) for _ in range(len(self.strides))]
        return BufferList(cell_points)

    @property
    def num_points(self):
        return [1 for _ in self.strides]

    def _grid_points(self, grid_sizes: List[List[int]]):
        points = []
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_points.named_buffers()]
        for size, stride, base_point in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_point)
            shifts = torch.stack((shift_x, shift_y), dim=1)

            points.append((shifts.view(-1, 1, 2) + base_point.view(1, -1, 2)).reshape(-1, 2))

        return points

    def forward(self, features: List[torch.Tensor]):
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        points_over_all_feature_maps = self._grid_points(grid_sizes)
        return points_over_all_feature_maps
