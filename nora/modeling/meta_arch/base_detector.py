from .base import BaseModel

__all__ = ["Detector"]


class Detector(BaseModel):
    """
    Abstract base class for detector.
    """

    def __init__(self):
        super().__init__()

    @property
    def with_box(self) -> bool:
        return (hasattr(self, "box_head") and self.box_head is not None) or (hasattr(self, "roi_head") and self.roi_head.with_box)

    @property
    def with_mask(self) -> bool:
        return (hasattr(self, "mask_head") and self.mask_head is not None) or (hasattr(self, "roi_head") and self.roi_head.with_mask)

    @property
    def with_keypoint(self) -> bool:
        return (hasattr(self, "keypoint_head") and self.keypoint_head is not None) or (hasattr(self, "roi_head") and self.roi_head.with_keypoint)
