# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import List

from nora.utils.file_io import PathHandler
from nora.utils.file_io import PathManager


class Detectron2ModelCatalog:
    """
    Store mappings from names to detectron2 models.
    """

    PREFIX = ""

    IMAGENET_MODELS = {
    }

    COCO_DETECTION_MODELS = {
    }

    COCO_INSTANCE_SEGMENTATION_MODELS = {
    }

    @staticmethod
    def get(name: str):
        if name.startswith("ImageNet/"):
            return Detectron2ModelCatalog._get_imagenet_models(name)
        elif name.startswith("COCO-Detection/"):
            return Detectron2ModelCatalog._get_coco_detection_models(name)
        elif name.startswith("COCO-InstanceSegmentation/"):
            return Detectron2ModelCatalog._get_coco_instance_segmentation_models(name)
        else:
            raise RuntimeError(f"model not present in detectron2 catalog: {name}")

    @staticmethod
    def _get_imagenet_models(name: str):
        prefix = Detectron2ModelCatalog.PREFIX
        model_name = name[len("ImageNet/"):]
        path = Detectron2ModelCatalog.IMAGENET_MODELS[model_name]
        return "".join([prefix, "ImageNet/", path])

    @staticmethod
    def _get_coco_detection_models(name: str):
        prefix = Detectron2ModelCatalog.PREFIX
        model_name = name[len("COCO-Detection/"):]
        path = Detectron2ModelCatalog.COCO_DETECTION_MODELS[model_name]
        return "".join([prefix, "COCO-Detection/", path])

    @staticmethod
    def _get_coco_instance_segmentation_models(name: str):
        prefix = Detectron2ModelCatalog.PREFIX
        model_name = name[len("COCO-InstanceSegmentation/"):]
        path = Detectron2ModelCatalog.COCO_INSTANCE_SEGMENTATION_MODELS[model_name]
        return "".join([prefix, "COCO-InstanceSegmentation/", path])


class TorchvisionModelCatalog:
    """
    Store mappings from names to torchvision models.
    """

    PREFIX = ""

    IMAGENET_MODELS = {
    }

    @staticmethod
    def get(name: str):
        if name.startswith("ImageNet/"):
            return TorchvisionModelCatalog._get_imagenet_models(name)
        else:
            raise RuntimeError(f"model not present in torchvision catalog: {name}")

    @staticmethod
    def _get_imagenet_models(name: str):
        prefix = TorchvisionModelCatalog.PREFIX
        model_name = name[len("ImageNet/"):]
        path = TorchvisionModelCatalog.IMAGENET_MODELS[model_name]
        return "".join([prefix, "ImageNet/", path])


class TIMMModelCatalog:
    """
    Store mappings from names to timm models.
    """

    PREFIX = ""

    IMAGENET_MODELS = {
    }

    @staticmethod
    def get(name: str):
        if name.startswith("ImageNet/"):
            return TIMMModelCatalog._get_imagenet_models(name)
        else:
            raise RuntimeError(f"model not present in timm catalog: {name}")

    @staticmethod
    def _get_imagenet_models(name: str):
        prefix = TIMMModelCatalog.PREFIX
        model_name = name[len("ImageNet/"):]
        path = TIMMModelCatalog.IMAGENET_MODELS[model_name]
        return "".join([prefix, "ImageNet/", path])


class OpenMMLabModelCatalog:
    """
    Store mappings from names to openmmlab models
    """

    PREFIX = ""

    MMPRETRAIN_MODELS = {
    }

    MMDET_MODELS = {
    }

    MMYOLO_MODELS = {
        "RTMDet-large": "rtmdet_large_300e-1d35e798.pth",
        "RTMDet-medium": "rtmdet_medium_300e-a511f6c8.pth",
        "RTMDet-small": "rtmdet_small_300e-f63fd229.pth",
        "RTMDet-tiny": "rtmdet_tiny_300e-1f49566c.pth",
    }

    MMSEG_MODELS = {
    }

    MMRAZOR_MODELS = {
    }

    MMPOSE_MODELS = {
    }

    @staticmethod
    def get(name: str):
        if name.startswith("mmpretrain/"):
            return OpenMMLabModelCatalog._get_mmpretrain_models(name)
        elif name.startswith("mmdet/"):
            return OpenMMLabModelCatalog._get_mmdet_models(name)
        elif name.startswith("mmyolo/"):
            return OpenMMLabModelCatalog._get_mmyolo_models(name)
        elif name.startswith("mmseg/"):
            return OpenMMLabModelCatalog._get_mmseg_models(name)
        elif name.startswith("mmrazor/"):
            return OpenMMLabModelCatalog._get_mmrazor_models(name)
        elif name.startswith("mmpose/"):
            return OpenMMLabModelCatalog._get_mmpose_models(name)
        else:
            raise RuntimeError(f"model not present in openmmlab catalog: {name}")

    @staticmethod
    def _get_mmpretrain_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmpretrain/"):]
        path = OpenMMLabModelCatalog.MMPRETRAIN_MODELS[model_name]
        return "".join([prefix, "mmpretrain/", path])

    @staticmethod
    def _get_mmdet_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmdet/"):]
        path = OpenMMLabModelCatalog.MMDET_MODELS[model_name]
        return "".join([prefix, "mmdet/", path])

    @staticmethod
    def _get_mmyolo_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmyolo/"):]
        path = OpenMMLabModelCatalog.MMYOLO_MODELS[model_name]
        return "".join([prefix, "mmyolo/", path])

    @staticmethod
    def _get_mmseg_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmseg/"):]
        path = OpenMMLabModelCatalog.MMSEG_MODELS[model_name]
        return "".join([prefix, "mmseg/", path])

    @staticmethod
    def _get_mmrazor_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmrazor/"):]
        path = OpenMMLabModelCatalog.MMRAZOR_MODELS[model_name]
        return "".join([prefix, "mmrazor/", path])

    @staticmethod
    def _get_mmpose_models(name: str):
        prefix = OpenMMLabModelCatalog.PREFIX
        model_name = name[len("mmpose/"):]
        path = OpenMMLabModelCatalog.MMPOSE_MODELS[model_name]
        return "".join([prefix, "mmpose/", path])


class Detectron2PathHandler(PathHandler):
    """
    Resolve anything that's hosted under detectron2's namespace.
    """

    PREFIX = "detectron2://"

    def get_supported_prefixes(self) -> List[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        name = path[len(self.PREFIX):]
        catalog_path = Detectron2ModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        return PathManager.get_local_path(catalog_path, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


class TorchvisionPathHandler(PathHandler):
    """
    Resolve anything that's hosted under torchvision's namespace.
    """

    PREFIX = "torchvision://"

    def get_supported_prefixes(self) -> List[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        name = path[len(self.PREFIX):]
        catalog_path = TorchvisionModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        return PathManager.get_local_path(catalog_path, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


class TIMMPathHandler(PathHandler):
    """
    Resolve anything that's hosted under timm's namespace.
    """

    PREFIX = "timm://"

    def get_supported_prefixes(self) -> List[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        name = path[len(self.PREFIX):]
        catalog_path = TIMMModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        return PathManager.get_local_path(catalog_path, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


class OpenMMLabPathHandler(PathHandler):
    """
    Resolve anything that's hosted under openmmlab's namespace.
    """

    PREFIX = "openmmlab://"

    def get_supported_prefixes(self) -> List[str]:
        return [self.PREFIX]

    def get_local_path(self, path: str, **kwargs) -> str:
        logger = logging.getLogger(__name__)
        name = path[len(self.PREFIX):]
        catalog_path = OpenMMLabModelCatalog.get(name)
        logger.info(f"Catalog entry {path} points to {catalog_path}")
        return PathManager.get_local_path(catalog_path, **kwargs)

    def open(self, path, mode="r", **kwargs):
        return PathManager.open(self.get_local_path(path), mode, **kwargs)


PathManager.register_handler(Detectron2PathHandler())
PathManager.register_handler(TorchvisionPathHandler())
PathManager.register_handler(TIMMPathHandler())
PathManager.register_handler(OpenMMLabPathHandler())
