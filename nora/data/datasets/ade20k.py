from nora.data.catalog import DatasetCatalog
from nora.data.catalog import MetadataCatalog
from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES
from .coco import load_sem_seg

__all__ = ["register_ade20k_sem_seg"]


def register_ade20k_sem_seg(name, image_dir, gt_dir):
    DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg"))
    MetadataCatalog.get(name).set(
        stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=255,
    )
