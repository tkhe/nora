# Copyright (c) Facebook, Inc. and its affiliates.

import os

from nora.checkpoint import Checkpointer
from nora.config import LazyConfig
from nora.data import MetadataCatalog
from nora.engine import DefaultTrainer
from nora.engine import default_argument_parser
from nora.engine import default_setup
from nora.engine import launch
from nora.evaluation import CityscapesInstanceEvaluator
from nora.evaluation import CityscapesSemSegEvaluator
from nora.evaluation import COCOEvaluator
from nora.evaluation import COCOPanopticEvaluator
from nora.evaluation import DatasetEvaluators
from nora.evaluation import LVISEvaluator
from nora.evaluation import PascalVOCDetectionEvaluator
from nora.evaluation import SemSegEvaluator


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.train.output_dir, "inference")

    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)

    if len(evaluator_list) == 0:
        raise NotImplementedError(f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}")
    elif len(evaluator_list) == 1:
        return evaluator_list[0]

    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
        res = Trainer.test(cfg, model)

        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
