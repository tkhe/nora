import torch
from torchvision.ops import PSRoIAlign
from torchvision.ops import PSRoIPool
from torchvision.ops import RoIAlign
from torchvision.ops import RoIPool

__all__ = [
    "PSROIAlign",
    "PSROIPool",
    "ROIAlign",
    "ROIPool",
]


class PSROIAlign(PSRoIAlign):
    pass


class PSROIPool(PSRoIPool):
    pass


class ROIAlign(RoIAlign):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned: bool = True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super().__init__(output_size, spatial_scale, sampling_ratio, aligned)

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        assert rois.dim() == 2 and rois.size(1) == 5

        if input.is_quantized:
            input = input.dequantize()

        return super().forward(input, rois.to(dtype=input.dtype))


class ROIPool(RoIPool):
    pass
