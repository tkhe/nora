from .base_semantic_segmentor import SemanticSegmentor

__all__ = ["UPerNet"]


class UPerNet(SemanticSegmentor):
    """
    support UPerNet (https://arxiv.org/abs/1807.10221).
    """
