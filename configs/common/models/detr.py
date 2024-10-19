from nora.config import LazyCall as L
from nora.layers import ShapeSpec
from nora.layers import SinePositionEmbedding
from nora.modeling.backbone import ResNet
from nora.modeling.dense_head import DETRHead
from nora.modeling.loss import CrossEntropyLoss
from nora.modeling.loss import GIoULoss
from nora.modeling.loss import L1Loss
from nora.modeling.meta_arch import DETR
from nora.modeling.matcher import HungarianMatcher
from nora.modeling.matcher import CrossEntropyCost
from nora.modeling.matcher import GIoUCost
from nora.modeling.matcher import L1Cost
from nora.modeling.neck import ChannelMapper
from nora.modeling.transformer import DETRTransformer
from nora.modeling.transformer.detr_transformer import DETRTransformerDecoder
from nora.modeling.transformer.detr_transformer import DETRTransformerEncoder
from nora.model_zoo import get_config

constants = get_config("common/data/constants.py").constants

model = L(DETR)(
    backbone=L(ResNet.build)(
        name="ResNet-50",
        out_features=["res5"],
        freeze_at=1,
    ),
    neck=L(ChannelMapper)(
        input_shape={"res5": ShapeSpec(channels=2048, stride=32)},
        in_features="${..backbone.out_features}",
        out_channels=256,
    ),
    position_embedding=L(SinePositionEmbedding)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
    ),
    transformer=L(DETRTransformer)(
        in_features=["p5"],
        encoder=L(DETRTransformerEncoder)(
            num_layers=6,
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_channels=2048,
            ffn_dropout=0.1,
            batch_first=True,
        ),
        decoder=L(DETRTransformerDecoder)(
            num_layers=6,
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_channels=2048,
            ffn_dropout=0.1,
            return_intermediate=True,
            batch_first=True,
        ),
    ),
    box_head=L(DETRHead)(
        num_classes=80,
        embed_dim=256,
        matcher=L(HungarianMatcher)(
            match_costs=[
                L(CrossEntropyCost)(weight=1.0),
                L(GIoUCost)(weight=2.0),
                L(L1Cost)(weight=5.0),
            ],
        ),
        loss_cls=L(CrossEntropyLoss)(
            reduction="mean",
            loss_weight=1.0,
        ),
        loss_iou=L(GIoULoss)(
            reduction="sum",
            loss_weight=2.0,
        ),
        loss_reg=L(L1Loss)(
            reduction="sum",
            loss_weight=5.0,
        ),
        use_auxiliary_loss=True,
    ),
    num_queries=100,
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
)
