from nora.config import LazyCall as L
from nora.data import transforms as T
from nora.model_zoo import get_config
from nora.solver import MultiStepParamScheduler
from nora.solver import WarmupParamScheduler

model = get_config("common/models/conditional_detr.py").model

dataloader = get_config("common/data/coco_detection.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(prob=0.5),
    L(T.OneOf)(
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
            ),
            L(T.AugmentationList)(
                augs=[
                    L(T.ResizeShortestEdge)(short_edge_length=(400, 500, 600)),
                    L(T.RandomCrop)(
                        crop_type="absolute_range",
                        crop_size=(384, 600),
                        prob=1.0,
                    ),
                    L(T.ResizeShortestEdge)(
                        short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                        max_size=1333,
                        sample_style="choice",
                    ),
                ],
            ),
        ],
        prob=1.0
    ),
]
dataloader.train.total_batch_size = 64
dataloader.train.num_workers = 16

optimizer = get_config("common/optim.py").AdamW
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda x: 0.1 if "backbone" in x else 1

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[75000, 93750]
    ),
)

train = get_config("common/train.py").train
train.max_iter = 93750
train.init_checkpoint = "torchvision://ImageNet/ResNet-50"
