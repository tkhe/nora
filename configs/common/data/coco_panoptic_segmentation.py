from .coco_instance_segmentation import dataloader

dataloader.train.dataset.names = ["coco_2017_train_panoptic_separated"]
dataloader.train.dataset.filter_empty = False

dataloader.test.dataset.names = ["coco_2017_val_panoptic_separated"]
