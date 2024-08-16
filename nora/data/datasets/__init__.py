from . import builtin as _builtin  # ensure the builtin datasets are registered
from .ade20k import *
from .cityscapes import *
from .cityscapes_panoptic import *
from .coco import *
from .coco_panoptic import *
from .lvis import *
from .pascal_voc import *

__all__ = [k for k in globals().keys() if not k.startswith("_")]
