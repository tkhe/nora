from nora.data.detection_utils import create_keypoint_hflip_indices
from .coco_detection import dataloader

dataloader.train.dataset.min_keypoints = 1
dataloader.train.dataset.names = ["keypoints_coco_2017_train"]
dataloader.train.mapper.use_keypoint = True
dataloader.train.mapper.keypoint_hflip_indices = create_keypoint_hflip_indices(dataloader.train.dataset.names)

dataloader.test.dataset.names = ["keypoints_coco_2017_val"]
