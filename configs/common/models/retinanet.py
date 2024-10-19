from nora.config import LazyCall as L
from nora.layers import ShapeSpec
from nora.modeling.anchor_generator import AnchorGenerator
from nora.modeling.backbone import ResNet
from nora.modeling.box_coder import DeltaXYWHBoxCoder
from nora.modeling.dense_head import RetinaNetHead
from nora.modeling.loss import SigmoidFocalLoss
from nora.modeling.loss import SmoothL1Loss
from nora.modeling.matcher import MaxIoUMatcher
from nora.modeling.meta_arch import RetinaNet
from nora.modeling.neck import FPN
from nora.modeling.neck.fpn import LastLevelP6P7
from nora.model_zoo import get_config

constants = get_config("common/data/constants.py").constants

model = L(RetinaNet)(
    backbone=L(ResNet.build)(
        name="ResNet-50",
        stride_in_1x1=True,
        norm="FrozenBN",
        out_features=["res3", "res4", "res5"],
        freeze_at=2,
    ),
    neck=L(FPN)(
        input_shape={
            "res3": ShapeSpec(channels=512, stride=8),
            "res4": ShapeSpec(channels=1024, stride=16),
            "res5": ShapeSpec(channels=2048, stride=32),
        },
        in_features="${..backbone.out_features}",
        out_channels=256,
        top_block=L(LastLevelP6P7)(
            in_channels=2048,
            out_channels="${..out_channels}",
            in_feature="res5",
        ),
    ),
    box_head=L(RetinaNetHead)(
        input_shape={
            "p3": ShapeSpec(channels=256, stride=8),
            "p4": ShapeSpec(channels=256, stride=16),
            "p5": ShapeSpec(channels=256, stride=32),
            "p6": ShapeSpec(channels=256, stride=64),
            "p7": ShapeSpec(channels=256, stride=128),
        },
        in_features=["p3", "p4", "p5", "p6", "p7"],
        num_classes=80,
        conv_dims=[256, 256, 256, 256],
        anchor_generator=L(AnchorGenerator)(
            sizes=[[x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)] for x in [32, 64, 128, 256, 512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
            offset=0,
        ),
        box_coder=L(DeltaXYWHBoxCoder)(weights=[1.0, 1.0, 1.0, 1.0]),
        matcher=L(MaxIoUMatcher)(
            thresholds=[0.4, 0.5],
            labels=[0, -1, 1],
            allow_low_quality_matches=True,
        ),
        loss_cls=L(SigmoidFocalLoss)(
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ),
        loss_reg=L(SmoothL1Loss)(
            beta=0,
            reduction="sum",
        ),
        test_score_threshold=0.05,
        test_topk_candidates=1000,
        test_nms_threshold=0.5,
        max_detections_per_image=100,
    ),
    pixel_mean=constants.imagenet_bgr256_mean,
    pixel_std=[1.0, 1.0, 1.0],
)
