from nora.config import LazyCall as L
from nora.modeling.backbone import ResNet
from nora.modeling.backbone.resnet import DeepStem
from nora.modeling.backbone.resnet import Bottleneck
from nora.modeling.loss import CrossEntropyLoss
from nora.modeling.meta_arch import UPerNet
from nora.modeling.sem_seg_head import UPerNetHead
from nora.model_zoo import get_config

constants = get_config("common/data/constants.py").constants

model = L(UPerNet)(
    backbone=L(ResNet)(
        stem_class=DeepStem,
        block_class=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        out_channels_per_stage=[256, 512, 1024, 2048],
        norm="FrozenBN",
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=2,
    ),
    sem_seg_head=L(UPerNetHead)(
        in_features="${..backbone.out_features}",
        num_classes=150,
        channels=512,
        pool_scales=[1, 2, 3, 6],
        dropout=0.1,
        norm="SyncBN",
        align_corners=False,
        loss_sem_seg=L(CrossEntropyLoss)(loss_weight=1.0),
    ),
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
