from nora.model_zoo import get_config
from .retinanet_resnet50_fpn_1x import model
from .retinanet_resnet50_fpn_1x import dataloader
from .retinanet_resnet50_fpn_1x import optimizer
from .retinanet_resnet50_fpn_1x import train

lr_multiplier = get_config("common/scheduler/coco_scheduler.py").lr_multiplier_3x
