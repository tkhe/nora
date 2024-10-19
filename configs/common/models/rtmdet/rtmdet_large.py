from nora.config import LazyCall as L
from nora.layers import ShapeSpec
from nora.modeling.anchor_generator import PointGenerator
from nora.modeling.backbone import CSPNeXt
from nora.modeling.box_coder import DistancePointBoxCoder
from nora.modeling.dense_head import RTMDetHead
from nora.modeling.loss import GIoULoss
from nora.modeling.loss import QualityFocalLoss
from nora.modeling.matcher import DynamicSoftLabelMatcher
from nora.modeling.meta_arch import RTMDet
from nora.modeling.neck import CSPNeXtPAFPN
from nora.model_zoo import get_config

constants = get_config("common/data/constants.py").constants

deepen_factor = 1.0
widen_factor = 1.0

model = L(RTMDet)(
    backbone=L(CSPNeXt.build)(
        name="CSPNeXt-large",
        out_features=["stage3", "stage4", "stage5"],
        freeze_at=-1,
    ),
    neck=L(CSPNeXtPAFPN)(
        input_shape={
            "stage3": ShapeSpec(channels=256, stride=8),
            "stage4": ShapeSpec(channels=512, stride=16),
            "stage5": ShapeSpec(channels=1024, stride=32),
        },
        in_features="${..backbone.out_features}",
        out_channels=256,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    box_head=L(RTMDetHead)(
        input_shape={
            "p3": ShapeSpec(channels=256, stride=8),
            "p4": ShapeSpec(channels=256, stride=16),
            "p5": ShapeSpec(channels=256, stride=32),
        },
        in_features=["p3", "p4", "p5"],
        num_classes=80,
        widen_factor=widen_factor,
        feat_channels=256,
        stacked_convs=2,
        share_conv=True,
        point_generator=L(PointGenerator)(
            strides=[8, 16, 32],
            offset=0,
        ),
        box_coder=L(DistancePointBoxCoder)(),
        matcher=L(DynamicSoftLabelMatcher)(topk=13),
        loss_cls=L(QualityFocalLoss)(
            reduction="sum",
            loss_weight=1.0,
        ),
        loss_reg=L(GIoULoss)(
            reduction="none",
            loss_weight=2.0,
        ),
        test_score_threshold=0.05,
        test_topk_candidates=1000,
        test_nms_threshold=0.5,
        max_detections_per_image=100,
    ),
    pixel_mean=constants.imagenet_bgr256_mean,
    pixel_std=constants.imagenet_bgr256_std,
)
