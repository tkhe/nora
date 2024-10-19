from .one_stage_detector import OneStageDetector

__all__ = ["RTMDet"]


class RTMDet(OneStageDetector):
    """
    support RTMDet (https://arxiv.org/abs/2212.07784).
    """
