# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import logging
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch

from . import transforms as T
from . import detection_utils as utils

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in nora dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        mask_format: str = "polygon",
        use_box: bool = True,
        use_mask: bool = False,
        use_keypoint: bool = False,
        use_sem_seg: bool = False,
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        recompute_boxes: bool = False,
    ):
        """
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            use_box: whether to process bounding box annotations, if available
            use_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            use_sem_seg: whether to process semantic segmentation annotations if available
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_mask, "recompute_boxes requires instance masks"

        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.mask_format = mask_format
        self.use_box = use_box
        self.use_mask = use_mask
        self.use_keypoint = use_keypoint
        self.use_sem_seg = use_sem_seg
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.recompute_boxes = recompute_boxes

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.mask_format)

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in nora dataset format.

        Returns:
            dict: a format that builtin models in nora accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if self.use_sem_seg:
            assert "sem_seg_file_name" in dataset_dict

            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
