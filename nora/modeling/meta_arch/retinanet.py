from .one_stage_detector import OneStageDetector

__all__ = ["RetinaNet"]


class RetinaNet(OneStageDetector):
    """
    support RetinaNet (https://arxiv.org/abs/1708.02002).
    """
