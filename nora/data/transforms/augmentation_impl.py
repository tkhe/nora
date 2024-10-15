# Copyright (c) Facebook, Inc. and its affiliates.

import sys
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from .augmentation import Augmentation
from .augmentation import _transform_to_aug
from .transform import CropTransform
from .transform import HorizontalFlipTransform
from .transform import NoOpTransform
from .transform import PadTransform
from .transform import ResizeTransform
from .transform import RotationTransform
from .transform import Transform
from .transform import TransformList
from .transform import VerticalFlipTransform

__all__ = [
    "FixedSizeCrop",
    "OneOf",
    "RandomApply",
    "RandomCrop",
    "RandomFlip",
    "RandomRotation",
    "Resize",
    "ResizeScale",
    "ResizeShortestEdge",
]


class RandomApply(Augmentation):
    """
    Randomly apply an augmentation with a given probability.
    """

    def __init__(self, tfm_or_aug: Union[Augmentation, Transform], prob: float = 0.5):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()

        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"

        self.aug = _transform_to_aug(tfm_or_aug)
        self.prob = prob

    def get_transform(self, *args) -> Transform:
        do = self._rand_range() < self.prob
        if do:
            return self.aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input) -> Transform:
        do = self._rand_range() < self.prob
        if do:
            return self.aug(aug_input)
        else:
            return NoOpTransform()


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob: float = 0.5, direction: str = "horizontal"):
        """
        Args:
            prob (float): probability of flip.
            direction (str): whether to apply horizontal or vertical flipping
        """
        super().__init__()

        assert direction in ("horizontal", "vertical")

        self.prob = prob
        self.direction = direction

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[:2]

        do = self._rand_range() < self.prob
        if do:
            if self.direction == "horizontal":
                return HorizontalFlipTransform(w)
            elif self.direction == "vertical":
                return VerticalFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(Augmentation):
    """
    Resize image to a fixed target size.
    """

    def __init__(self, shape: Union[int, Tuple[int, int]], interp: str = "bilinear"):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: interpolation method
        """
        super().__init__()

        if isinstance(shape, int):
            shape = (shape, shape)

        shape = tuple(shape)

        self.shape = shape
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        return ResizeTransform(image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp)


class ResizeShortestEdge(Augmentation):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(self, short_edge_length: List[int], max_size: int = sys.maxsize, sample_style: str = "choice", interp: str = "bilinear"):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"

        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)

        if self.is_range:
            assert len(short_edge_length) == 2, f"short_edge_length must be two values using 'range' sample style. Got {short_edge_length}!"

        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.sample_style = sample_style
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[:2]

        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)

        if size == 0:
            return NoOpTransform()

        newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
        return ResizeTransform(h, w, newh, neww, self.interp)

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class RandomCrop(Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size: Tuple[float, float], prob: float = 0.5):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.

        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        assert crop_type in ("relative_range", "relative", "absolute", "absolute_range")

        self.crop_type = crop_type
        self.crop_size = crop_size
        self.prob = prob

    def get_transform(self, image: np.ndarray) -> Transform:
        do = self._rand_range() < self.prob

        if do:
            h, w = image.shape[:2]
            croph, cropw = self.get_crop_size((h, w))

            assert h >= croph and w >= cropw, f"Shape computation in {self} has bugs."

            h0 = np.random.randint(h - croph + 1)
            w0 = np.random.randint(w - cropw + 1)
            return CropTransform(w0, h0, cropw, croph)
        else:
            return NoOpTransform()

    def get_crop_size(self, image_size: Tuple[int, int]):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size

        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]

            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError(f"Unknown crop type {self.crop_type}")


class ResizeScale(Augmentation):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interp: str = "bilinear",
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        super().__init__()

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_height = target_height
        self.target_width = target_width
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        random_scale = np.random.uniform(self.min_scale, self.max_scale)
        input_size = image.shape[:2]

        # Compute new target size given a scale.
        target_size = (self.target_height, self.target_width)
        target_scale_size = np.multiply(target_size, random_scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1])
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

        return ResizeTransform(input_size[0], input_size[1], int(output_size[0]), int(output_size[1]), self.interp)


class FixedSizeCrop(Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """

    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]],
        pad: bool = True,
        pad_value: float = 114,
        seg_pad_value: int = 255,
    ):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value to the image.
            seg_pad_value: the padding value to the segmentation mask.
        """
        super().__init__()

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        self.crop_size = crop_size
        self.pad = pad
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value

    def _get_crop(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        return CropTransform(offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0])

    def _get_pad(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        original_size = np.minimum(input_size, output_size)
        return PadTransform(
            0,
            0,
            pad_size[1],
            pad_size[0],
            original_size[1],
            original_size[0],
            self.pad_value,
            self.seg_pad_value,
        )

    def get_transform(self, image: np.ndarray) -> TransformList:
        transforms = [self._get_crop(image)]
        if self.pad:
            transforms.append(self._get_pad(image))
        return TransformList(transforms)


class RandomRotation(Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(
        self,
        angle: List[float],
        expand: bool = True,
        center: List[Tuple[float, float]]=None,
        sample_style: str = "range",
        interp: str = "bilinear",
    ):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()

        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)

        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)

        self.angle = angle
        self.center = center
        self.expand = expand
        self.interp = interp

    def get_transform(self, image) -> Transform:
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class OneOf(Augmentation):
    def __init__(
        self,
        augmentations: List[Augmentation],
        prob: float = 0.5,
    ):
        super().__init__()

        self.augmentations = augmentations
        self.prob = prob

    def __call__(self, aug_input):
        do = self._rand_range() < self.prob
        if do:
            aug = np.random.choice(self.augmentations, 1)[0]
            tfm = aug(aug_input)
        else:
            tfm = NoOpTransform()

        return tfm
